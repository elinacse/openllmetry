from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Collection, Tuple, cast, Optional
import json
import logging
import traceback
import re
import time
from http import HTTPStatus

from opentelemetry import context, propagate
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.trace import get_tracer, Tracer
from wrapt import ObjectProxy, register_post_import_hook, wrap_function_wrapper
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.semconv_ai import SpanAttributes
from opentelemetry.semconv.attributes.error_attributes import ERROR_TYPE

from opentelemetry.instrumentation.mcp.version import __version__

_instruments = ("mcp >= 1.6.0",)


class Config:
    exception_logger = None


class McpSessionInfo:
    """Stores MCP session information for span attribution"""
    transport_mode = None
    client_id = None
    client_name = None
    protocol_version = None
    response_chunk_count = 0
    
    @classmethod
    def set_transport_mode(cls, mode: str):
        cls.transport_mode = mode
    
    @classmethod
    def set_client_info(cls, client_id: Optional[str] = None, client_name: Optional[str] = None):
        if client_id is not None:
            cls.client_id = client_id
        if client_name is not None:
            cls.client_name = client_name
    
    @classmethod
    def set_protocol_version(cls, version: str):
        cls.protocol_version = version
    
    @classmethod
    def increment_chunk_count(cls):
        cls.response_chunk_count += 1
    
    @classmethod
    def reset_chunk_count(cls):
        cls.response_chunk_count = 0
    
    @classmethod
    def get_chunk_count(cls):
        return cls.response_chunk_count
    
    @classmethod
    def add_session_attributes(cls, span):
        """Add session attributes to a span"""
        if cls.client_id:
            span.set_attribute(SpanAttributes.MCP_CLIENT_ID, cls.client_id)
        if cls.client_name:
            span.set_attribute(SpanAttributes.MCP_CLIENT_NAME, cls.client_name)
        if cls.transport_mode:
            span.set_attribute(SpanAttributes.MCP_TRANSPORT_MODE, cls.transport_mode)
        if cls.protocol_version:
            span.set_attribute(SpanAttributes.MCP_PROTOCOL_VERSION, cls.protocol_version)


def dont_throw(func):
    """
    A decorator that wraps the passed in function and logs exceptions instead of throwing them.

    @param func: The function to wrap
    @return: The wrapper function
    """
    # Obtain a logger specific to the function's module
    logger = logging.getLogger(func.__module__)

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.debug(
                "OpenLLMetry failed to trace in %s, error: %s",
                func.__name__,
                traceback.format_exc(),
            )
            if Config.exception_logger:
                Config.exception_logger(e)

    return wrapper


class McpInstrumentor(BaseInstrumentor):
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs):
        tracer_provider = kwargs.get("tracer_provider")
        tracer = get_tracer(__name__, __version__, tracer_provider)

        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.sse", "sse_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.sse",
                "SseServerTransport.connect_sse",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.sse",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.stdio", "stdio_client", self._transport_wrapper(tracer)
            ),
            "mcp.client.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.stdio", "stdio_server", self._transport_wrapper(tracer)
            ),
            "mcp.server.stdio",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.session",
                "ServerSession.__init__",
                self._base_session_init_wrapper(tracer),
            ),
            "mcp.server.session",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.client.streamable_http",
                "streamablehttp_client",
                self._transport_wrapper(tracer),
            ),
            "mcp.client.streamable_http",
        )
        register_post_import_hook(
            lambda _: wrap_function_wrapper(
                "mcp.server.streamable_http",
                "StreamableHTTPServerTransport.connect",
                self._transport_wrapper(tracer),
            ),
            "mcp.server.streamable_http",
        )
        wrap_function_wrapper(
            "mcp.shared.session",
            "BaseSession.send_request",
            self.patch_mcp_client(tracer),
        )

    def _uninstrument(self, **kwargs):
        unwrap("mcp.client.stdio", "stdio_client")
        unwrap("mcp.server.stdio", "stdio_server")

    def _transport_wrapper(self, tracer):
        @asynccontextmanager
        async def traced_method(
            wrapped: Callable[..., Any], instance: Any, args: Any, kwargs: Any
        ) -> AsyncGenerator[
            Tuple["InstrumentedStreamReader", "InstrumentedStreamWriter"], None
        ]:
            # Detect transport mode from the function name
            func_name = getattr(wrapped, "__name__", "")
            module_name = getattr(wrapped, "__module__", "")
            
            if "stdio" in func_name or "stdio" in module_name:
                McpSessionInfo.set_transport_mode("stdio")
            elif "streamable" in func_name or "streamable" in module_name:
                McpSessionInfo.set_transport_mode("streamable-http")
            elif "sse" in func_name or "sse" in module_name:
                McpSessionInfo.set_transport_mode("websocket")
            elif "websocket" in func_name or "websocket" in module_name:
                McpSessionInfo.set_transport_mode("websocket")
            
            async with wrapped(*args, **kwargs) as result:
                try:
                    read_stream, write_stream = result
                except ValueError:
                    read_stream, write_stream, _ = result
                yield InstrumentedStreamReader(
                    read_stream, tracer
                ), InstrumentedStreamWriter(write_stream, tracer)

        return traced_method

    def _base_session_init_wrapper(self, tracer):
        def traced_method(
            wrapped: Callable[..., None], instance: Any, args: Any, kwargs: Any
        ) -> None:
            wrapped(*args, **kwargs)
            
            # Try to extract client information from the session
            # Generate a unique client ID if not available
            import uuid
            client_id = str(uuid.uuid4())
            McpSessionInfo.set_client_info(client_id=client_id)
            
            # Try to detect client name from environment or session attributes
            client_name = self._detect_client_name(instance, args, kwargs)
            if client_name:
                McpSessionInfo.set_client_info(client_name=client_name)
            
            reader = getattr(instance, "_incoming_message_stream_reader", None)
            writer = getattr(instance, "_incoming_message_stream_writer", None)
            if reader and writer:
                setattr(
                    instance,
                    "_incoming_message_stream_reader",
                    ContextAttachingStreamReader(reader, tracer),
                )
                setattr(
                    instance,
                    "_incoming_message_stream_writer",
                    ContextSavingStreamWriter(writer, tracer),
                )

        return traced_method

    def _detect_client_name(self, instance, args, kwargs):
        """Attempt to detect client name from various sources"""
        import os
        
        # Check environment variables that might contain client info
        client_name = os.environ.get("MCP_CLIENT_NAME")
        if client_name:
            return client_name
            
        # Check for common client indicators in environment
        if os.environ.get("CLAUDE_DESKTOP"):
            return "Claude Desktop"
        elif os.environ.get("GITHUB_COPILOT"):
            return "GitHub Copilot"
        elif os.environ.get("STREAMLIT"):
            return "Streamlit UI"
        elif "streamlit" in os.environ.get("_", "").lower():
            return "Streamlit UI"
        
        # Check user agent or similar headers if available
        user_agent = os.environ.get("HTTP_USER_AGENT", "")
        if "claude" in user_agent.lower():
            return "Claude Desktop"
        elif "copilot" in user_agent.lower():
            return "GitHub Copilot"
        
        # Try to extract from instance attributes
        if hasattr(instance, "client_name"):
            return getattr(instance, "client_name")
        elif hasattr(instance, "name"):
            return getattr(instance, "name")
        
        # Check process name as fallback
        try:
            import psutil
            current_process = psutil.Process()
            process_name = current_process.name()
            if "claude" in process_name.lower():
                return "Claude Desktop"
            elif "copilot" in process_name.lower():
                return "GitHub Copilot"
        except ImportError:
            pass
        
        return "Unknown Client"

    def _detect_protocol_version(self, args):
        """Attempt to detect MCP protocol version from request"""
        try:
            if len(args) > 0 and hasattr(args[0], "root"):
                request = args[0].root
                
                # Check for version in the request headers or metadata
                if hasattr(request, "jsonrpc"):
                    jsonrpc_version = getattr(request, "jsonrpc", None)
                    if jsonrpc_version:
                        McpSessionInfo.set_protocol_version(f"JSON-RPC {jsonrpc_version}")
                
                # Check for MCP-specific version information
                if hasattr(request, "params") and request.params:
                    if hasattr(request.params, "version"):
                        version = getattr(request.params, "version", None)
                        if version:
                            McpSessionInfo.set_protocol_version(str(version))
                    elif hasattr(request.params, "__dict__") and "version" in request.params.__dict__:
                        version = request.params.__dict__["version"]
                        if version:
                            McpSessionInfo.set_protocol_version(str(version))
                
                # Fallback: try to determine from MCP library version
                try:
                    import mcp
                    if hasattr(mcp, "__version__"):
                        McpSessionInfo.set_protocol_version(f"MCP {mcp.__version__}")
                except ImportError:
                    pass
                    
                # If still no version, set a default
                if not McpSessionInfo.protocol_version:
                    McpSessionInfo.set_protocol_version("MCP 1.0")
                    
        except Exception:
            # Fallback version if detection fails
            if not McpSessionInfo.protocol_version:
                McpSessionInfo.set_protocol_version("MCP 1.0")

    def patch_mcp_client(self, tracer: Tracer):
        @dont_throw
        async def traced_method(wrapped, instance, args, kwargs):
            # Record request start time
            request_start_time = time.time()
            request_start_timestamp = time.time_ns() // 1_000_000  # Convert to milliseconds
            
            meta = None
            method = None
            params = None
            if len(args) > 0 and hasattr(args[0].root, "method"):
                method = args[0].root.method
            if len(args) > 0 and hasattr(args[0].root, "params"):
                params = args[0].root.params
            if params:
                if hasattr(args[0].root.params, "meta"):
                    meta = args[0].root.params.meta

            # Determine if this is a tool call
            is_tool_call = method == "tools/call"
            tool_name = None
            tool_args = None
            
            if is_tool_call and params:
                # Extract tool name and arguments for tool calls
                if hasattr(params, "name"):
                    tool_name = params.name
                elif hasattr(params, "__dict__") and "name" in params.__dict__:
                    tool_name = params.__dict__["name"]
                
                if hasattr(params, "arguments"):
                    tool_args = params.arguments
                elif hasattr(params, "__dict__") and "arguments" in params.__dict__:
                    tool_args = params.__dict__["arguments"]

            span_name = f"mcp.tool.{tool_name}" if is_tool_call and tool_name else f"{method}.mcp"
            
            # Try to detect protocol version from the request
            self._detect_protocol_version(args)
            
            with tracer.start_as_current_span(span_name) as span:
                # Add timing attributes
                span.set_attribute(SpanAttributes.MCP_REQUEST_START_TIME, str(request_start_timestamp))
                
                # Calculate and add request size
                request_size = calculate_payload_size(args[0]) if len(args) > 0 else 0
                span.set_attribute(SpanAttributes.MCP_REQUEST_SIZE_BYTES, str(request_size))
                
                span.set_attribute(
                    SpanAttributes.TRACELOOP_ENTITY_INPUT, f"{serialize(args[0])}"
                )
                
                # Add session attributes to all spans
                McpSessionInfo.add_session_attributes(span)
                
                # Add tool-specific attributes for tool calls
                if is_tool_call:
                    if tool_name:
                        span.set_attribute(SpanAttributes.MCP_TOOL_NAME, tool_name)
                    if tool_args:
                        span.set_attribute(SpanAttributes.MCP_TOOL_ARGS, str(serialize(tool_args)))

                if meta and len(args) > 0:
                    carrier = {}
                    TraceContextTextMapPropagator().inject(carrier)
                    meta.traceparent = carrier["traceparent"]
                    args[0].root.params.meta = meta
                try:
                    # Record tool execution start time for tool calls
                    tool_execution_start_time = None
                    if is_tool_call:
                        tool_execution_start_time = time.time()
                    
                    result = await wrapped(*args, **kwargs)
                    
                    # Calculate and record timing information
                    request_end_time = time.time()
                    request_end_timestamp = time.time_ns() // 1_000_000  # Convert to milliseconds
                    request_duration_ms = (request_end_time - request_start_time) * 1000  # Convert to milliseconds
                    
                    # Add end time and duration attributes
                    span.set_attribute(SpanAttributes.MCP_REQUEST_END_TIME, str(request_end_timestamp))
                    span.set_attribute(SpanAttributes.MCP_REQUEST_DURATION_MS, str(int(request_duration_ms)))

                    # Add tool execution duration for tool calls
                    if is_tool_call and tool_execution_start_time is not None:
                        tool_execution_duration_ms = (request_end_time - tool_execution_start_time) * 1000
                        span.set_attribute(SpanAttributes.MCP_TOOL_EXECUTION_DURATION_MS, str(int(tool_execution_duration_ms)))

                    # Calculate and add response size
                    response_size = calculate_payload_size(result)
                    span.set_attribute(SpanAttributes.MCP_RESPONSE_SIZE_BYTES, str(response_size))

                    # Add tool output size for tool calls
                    if is_tool_call and result:
                        tool_output_size = calculate_payload_size(result)
                        span.set_attribute(SpanAttributes.MCP_TOOL_OUTPUT_SIZE_BYTES, str(tool_output_size))

                    span.set_attribute(
                        SpanAttributes.TRACELOOP_ENTITY_OUTPUT,
                        str(serialize(result)),
                    )
                    
                    # Handle tool call status and errors
                    if is_tool_call:
                        if hasattr(result, "isError") and result.isError:
                            span.set_attribute(SpanAttributes.MCP_TOOL_STATUS, "error")
                            if len(result.content) > 0:
                                error_message = result.content[0].text
                                # Truncate error message if too long
                                if len(error_message) > 500:
                                    error_message = error_message[:500] + "..."
                                span.set_attribute(SpanAttributes.MCP_TOOL_ERROR_MESSAGE, error_message)
                                span.set_status(Status(StatusCode.ERROR, error_message))
                                error_type = get_error_type(error_message)
                                if error_type is not None:
                                    span.set_attribute(ERROR_TYPE, error_type)
                                    span.set_attribute(SpanAttributes.MCP_TOOL_ERROR_TYPE, error_type)
                        else:
                            span.set_attribute(SpanAttributes.MCP_TOOL_STATUS, "success")
                            span.set_status(Status(StatusCode.OK))
                    else:
                        # Non-tool call handling
                        if hasattr(result, "isError") and result.isError:
                            if len(result.content) > 0:
                                span.set_status(
                                    Status(StatusCode.ERROR, f"{result.content[0].text}")
                                )
                                error_type = get_error_type(result.content[0].text)
                                if error_type is not None:
                                    span.set_attribute(ERROR_TYPE, error_type)
                        else:
                            span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    # Calculate timing even on exception
                    request_end_time = time.time()
                    request_end_timestamp = time.time_ns() // 1_000_000
                    request_duration_ms = (request_end_time - request_start_time) * 1000

                    span.set_attribute(SpanAttributes.MCP_REQUEST_END_TIME, str(request_end_timestamp))
                    span.set_attribute(SpanAttributes.MCP_REQUEST_DURATION_MS, str(int(request_duration_ms)))

                    # Add tool execution duration for tool calls even on exception
                    if is_tool_call and tool_execution_start_time is not None:
                        tool_execution_duration_ms = (request_end_time - tool_execution_start_time) * 1000
                        span.set_attribute(SpanAttributes.MCP_TOOL_EXECUTION_DURATION_MS, str(int(tool_execution_duration_ms)))

                    # Add response size for error responses (usually smaller)
                    error_response_size = len(str(e).encode('utf-8'))
                    span.set_attribute(SpanAttributes.MCP_RESPONSE_SIZE_BYTES, str(error_response_size))

                    # Handle exceptions for tool calls
                    if is_tool_call:
                        span.set_attribute(SpanAttributes.MCP_TOOL_STATUS, "error")
                        span.set_attribute(SpanAttributes.MCP_TOOL_ERROR_TYPE, type(e).__name__)
                        error_message = str(e)
                        if len(error_message) > 500:
                            error_message = error_message[:500] + "..."
                        span.set_attribute(SpanAttributes.MCP_TOOL_ERROR_MESSAGE, error_message)

                    span.set_attribute(ERROR_TYPE, type(e).__name__)
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return traced_method


def get_error_type(error_message):
    if not isinstance(error_message, str):
        return None
    match = re.search(r"\b(4\d{2}|5\d{2})\b", error_message)
    if match:
        num = int(match.group())
        if 400 <= num <= 599:
            return HTTPStatus(num).name
        else:
            return None
    else:
        return None


def calculate_payload_size(payload) -> int:
    """Calculate the size of a payload in bytes"""
    try:
        if isinstance(payload, str):
            return len(payload.encode('utf-8'))
        elif isinstance(payload, bytes):
            return len(payload)
        elif hasattr(payload, 'model_dump_json'):
            # Pydantic model
            json_str = payload.model_dump_json()
            return len(json_str.encode('utf-8'))
        elif hasattr(payload, '__dict__'):
            # Convert to JSON and measure
            json_str = serialize(payload)
            return len(json_str.encode('utf-8'))
        else:
            # Fallback: convert to string and measure
            json_str = json.dumps(payload) if hasattr(json, 'dumps') else str(payload)
            return len(json_str.encode('utf-8'))
    except Exception:
        # If size calculation fails, return 0
        return 0

def serialize(request, depth=0, max_depth=4):
    """Serialize input args to MCP server into JSON.
    The function accepts input object and converts into JSON
    keeping depth in mind to prevent creating large nested JSON"""
    if depth > max_depth:
        return {}
    depth += 1

    def is_serializable(request):
        try:
            json.dumps(request)
            return True
        except Exception:
            return False

    if is_serializable(request):
        return json.dumps(request)
    else:
        result = {}
        try:
            if hasattr(request, "model_dump_json"):
                return request.model_dump_json()
            if hasattr(request, "__dict__"):
                for attrib in request.__dict__:
                    if not attrib.startswith("_"):
                        if type(request.__dict__[attrib]) in [
                            bool,
                            str,
                            int,
                            float,
                            type(None),
                        ]:
                            result[str(attrib)] = request.__dict__[attrib]
                        else:
                            result[str(attrib)] = serialize(
                                request.__dict__[attrib], depth
                            )
        except Exception:
            pass
        return json.dumps(result)


class InstrumentedStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        from mcp.types import JSONRPCMessage, JSONRPCRequest
        from mcp.shared.message import SessionMessage

        async for item in self.__wrapped__:
            if isinstance(item, SessionMessage):
                request = cast(JSONRPCMessage, item.message).root
            elif type(item) is JSONRPCMessage:
                request = cast(JSONRPCMessage, item).root
            else:
                return
            if not isinstance(request, JSONRPCRequest):
                yield item
                continue

            if request.params:
                meta = request.params.get("_meta")
                if meta:
                    ctx = propagate.extract(meta)
                    restore = context.attach(ctx)
                    try:
                        yield item
                        continue
                    finally:
                        context.detach(restore)
            yield item


class InstrumentedStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        from mcp.types import JSONRPCMessage, JSONRPCRequest
        from mcp.shared.message import SessionMessage

        # Record streaming start time
        stream_start_time = time.time()

        if isinstance(item, SessionMessage):
            request = cast(JSONRPCMessage, item.message).root
        elif type(item) is JSONRPCMessage:
            request = cast(JSONRPCMessage, item).root
        else:
            return

        with self._tracer.start_as_current_span("ResponseStreamWriter") as span:
            try:
                # Calculate response size
                response_size = calculate_payload_size(item)
                span.set_attribute(SpanAttributes.MCP_RESPONSE_SIZE_BYTES, str(response_size))
                
                # Increment chunk count for each response
                McpSessionInfo.increment_chunk_count()
                chunk_count = McpSessionInfo.get_chunk_count()
                span.set_attribute(SpanAttributes.MCP_RESPONSE_CHUNK_COUNT, str(chunk_count))
                
                if hasattr(request, "result"):
                    span.set_attribute(
                        SpanAttributes.MCP_RESPONSE_VALUE, f"{serialize(request.result)}"
                    )
                    if "isError" in request.result:
                        if request.result["isError"] is True:
                            span.set_status(
                                Status(
                                    StatusCode.ERROR,
                                    f"{request.result['content'][0]['text']}",
                                )
                            )
                            error_type = get_error_type(
                                request.result["content"][0]["text"]
                            )
                            if error_type is not None:
                                span.set_attribute(ERROR_TYPE, error_type)
                if hasattr(request, "id"):
                    span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{request.id}")

                if not isinstance(request, JSONRPCRequest):
                    result = await self.__wrapped__.send(item)
                    # Calculate and add streaming duration
                    stream_end_time = time.time()
                    stream_duration_ms = (stream_end_time - stream_start_time) * 1000
                    span.set_attribute(SpanAttributes.MCP_STREAM_DURATION_MS, str(int(stream_duration_ms)))
                    return result
                    
                meta = None
                if not request.params:
                    request.params = {}
                meta = request.params.setdefault("_meta", {})

                propagate.get_global_textmap().inject(meta)
                result = await self.__wrapped__.send(item)
                
                # Calculate and add streaming duration
                stream_end_time = time.time()
                stream_duration_ms = (stream_end_time - stream_start_time) * 1000
                span.set_attribute(SpanAttributes.MCP_STREAM_DURATION_MS, str(int(stream_duration_ms)))
                
                return result
            except Exception as e:
                # Calculate streaming duration even on exception
                stream_end_time = time.time()
                stream_duration_ms = (stream_end_time - stream_start_time) * 1000
                span.set_attribute(SpanAttributes.MCP_STREAM_DURATION_MS, str(int(stream_duration_ms)))
                raise


@dataclass(frozen=True)
class ItemWithContext:
    item: Any
    ctx: context.Context


class ContextSavingStreamWriter(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    @dont_throw
    async def send(self, item: Any) -> Any:
        # Record streaming start time
        stream_start_time = time.time()
        
        with self._tracer.start_as_current_span("RequestStreamWriter") as span:
            try:
                # Calculate request size
                request_size = calculate_payload_size(item)
                span.set_attribute(SpanAttributes.MCP_REQUEST_SIZE_BYTES, str(request_size))
                
                if hasattr(item, "request_id"):
                    span.set_attribute(SpanAttributes.MCP_REQUEST_ID, f"{item.request_id}")
                if hasattr(item, "request"):
                    if hasattr(item.request, "root"):
                        if hasattr(item.request.root, "method"):
                            span.set_attribute(
                                SpanAttributes.MCP_METHOD_NAME,
                                f"{item.request.root.method}",
                            )
                        if hasattr(item.request.root, "params"):
                            span.set_attribute(
                                SpanAttributes.MCP_REQUEST_ARGUMENT,
                                f"{serialize(item.request.root.params)}",
                            )
                ctx = context.get_current()
                result = await self.__wrapped__.send(ItemWithContext(item, ctx))
                
                # Calculate and add streaming duration
                stream_end_time = time.time()
                stream_duration_ms = (stream_end_time - stream_start_time) * 1000
                span.set_attribute(SpanAttributes.MCP_STREAM_DURATION_MS, str(int(stream_duration_ms)))
                
                return result
            except Exception as e:
                # Calculate streaming duration even on exception
                stream_end_time = time.time()
                stream_duration_ms = (stream_end_time - stream_start_time) * 1000
                span.set_attribute(SpanAttributes.MCP_STREAM_DURATION_MS, str(int(stream_duration_ms)))
                raise


class ContextAttachingStreamReader(ObjectProxy):  # type: ignore
    # ObjectProxy missing context manager - https://github.com/GrahamDumpleton/wrapt/issues/73
    def __init__(self, wrapped, tracer):
        super().__init__(wrapped)
        self._tracer = tracer

    async def __aenter__(self) -> Any:
        return await self.__wrapped__.__aenter__()

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> Any:
        return await self.__wrapped__.__aexit__(exc_type, exc_value, traceback)

    async def __aiter__(self) -> AsyncGenerator[Any, None]:
        async for item in self.__wrapped__:
            item_with_context = cast(ItemWithContext, item)
            restore = context.attach(item_with_context.ctx)
            try:
                yield item_with_context.item
            finally:
                context.detach(restore)
