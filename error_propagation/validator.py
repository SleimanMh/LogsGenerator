def validate_payload(payload):
    """
    Real-world validation: ensure payload is a list of numbers.
    """
    if not isinstance(payload, list):
        raise TypeError(f"Expected payload list, got {type(payload).__name__}")

    if not payload:
        raise ValueError("Payload cannot be empty")

    # Ensure all values are numbers
    for item in payload:
        if not isinstance(item, (int, float)):
            raise ValueError(f"Invalid item '{item}' in payload. Must be numeric.")

    # Simulate shape issue: require at least 3 features
    if len(payload) < 3:
        raise ValueError(f"Payload must contain at least 3 features. Got {len(payload)}.")

    return payload
