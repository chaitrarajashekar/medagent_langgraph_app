import langwatch
from functools import wraps


def traced_node(node_name):
    """
    Wraps a LangGraph node in a LangWatch span.

    Root cause of "No span or trace found" error:
    langwatch.span() and add_evaluation() require an active parent trace.
    The parent trace is created by @langwatch.trace() in run_pipeline()
    in api.py. If nodes are called outside that context (e.g. run_local.py
    or unit tests), there is no active trace and get_current_span() fails.

    Fix applied here:
    1. Only call langwatch.span() if a parent trace is currently active
    2. All add_evaluation() and span.update() calls are wrapped in
       try/except so a tracing failure NEVER blocks the pipeline
    3. Falls back to running the node function unwrapped if no trace active
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if a parent trace is active before creating a child span
            # get_current_trace() returns None if no @langwatch.trace() is active
            active_trace = None
            try:
                active_trace = langwatch.get_current_trace()
            except Exception:
                pass

            if active_trace is None:
                # No parent trace — run node directly without any tracing
                # This happens in run_local.py, unit tests, or if LangWatch
                # is not set up. Pipeline still works normally.
                return func(*args, **kwargs)

            # Parent trace is active — wrap node in a child span
            try:
                with langwatch.span(name=node_name) as span:
                    state = args[0] if args else {}

                    # Safe input capture
                    try:
                        span.update(
                            input=str({
                                k: v for k, v in state.items()
                                if k in ["medications", "patient_age",
                                         "patient_conditions", "clinical_question"]
                            })[:400]
                        )
                    except Exception:
                        pass

                    result = func(*args, **kwargs)

                    # Safe output capture
                    try:
                        span.update(output=str(result)[:400])
                    except Exception:
                        pass

                    return result

            except Exception:
                # Span creation failed — still run the node, never block pipeline
                return func(*args, **kwargs)

        return wrapper

    return decorator


def safe_add_evaluation(span_or_trace, **kwargs):
    """
    Safely call add_evaluation() — swallows any LangWatch errors.
    Use this instead of direct span.add_evaluation() in nodes to
    prevent tracing failures from breaking the pipeline.

    Usage:
        safe_add_evaluation(
            langwatch.get_current_span(),
            name="G-IN: Input Validation",
            passed=True,
            is_guardrail=True,
            details="All checks passed",
        )
    """
    try:
        if span_or_trace is not None:
            span_or_trace.add_evaluation(**kwargs)
    except Exception:
        pass


def safe_span_update(span, **kwargs):
    """
    Safely call span.update() — swallows any LangWatch errors.
    """
    try:
        if span is not None:
            span.update(**kwargs)
    except Exception:
        pass


def get_safe_span():
    """
    Returns the current LangWatch span, or None if no trace is active.
    Always use this instead of langwatch.get_current_span() directly.
    """
    try:
        return langwatch.get_current_span()
    except Exception:
        return None