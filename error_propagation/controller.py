from error_propagation.service import process_user_request

def handle_request(request_data: dict):
    """
    Simulates a request handler. Validates and forwards 
    the request to the service layer.
    """
    user_id = request_data.get("user_id")
    payload = request_data.get("payload")

    # some real-world guard checks
    if user_id is None:
        raise ValueError("Missing user_id in request")

    return process_user_request(user_id, payload)
