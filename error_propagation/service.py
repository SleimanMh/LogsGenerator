from error_propagation.validator import validate_payload
from error_propagation.model_loader import load_model

def process_user_request(user_id: int, payload):
    """
    Represents business logic layer.
    Validates input and performs a ML model inference.
    """
    validated_data = validate_payload(payload)

    model = load_model()
    result = model.predict(validated_data)

    return {"user_id": user_id, "result": result}
