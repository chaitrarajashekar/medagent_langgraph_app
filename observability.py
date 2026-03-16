import langwatch
from functools import wraps


def traced_node(node_name):
    """
    Wraps a LangGraph node in a LangWatch span.
    Uses direct attribute assignment — confirmed working with
    langwatch.span methods: update, input, output.
    """

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                with langwatch.span(name=node_name) as span:
                    state  = args[0] if args else {}

                    # Set input before running
                    span.update(
                        input=str({
                            k: v for k, v in state.items()
                            if k in ["medications","patient_age",
                                     "patient_conditions","clinical_question"]
                        })[:300]
                    )

                    result = func(*args, **kwargs)

                    # Set output after running
                    span.update(
                        output=str(result)[:300]
                    )

                    return result

            except Exception:
                # Never block the pipeline — fall through without tracing
                return func(*args, **kwargs)

        return wrapper

    return decorator