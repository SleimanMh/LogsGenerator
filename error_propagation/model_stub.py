class DummyModel:
    """
    A stub representing an ML model.
    We intentionally create a real-world failure scenario:
    predicting on wrong-shaped input.
    """

    def predict(self, data):
        # Simulate shape mismatch issue (common in ML pipelines)
        if len(data) != 3:
            raise ValueError(
                f"Model expected exactly 3 features, got {len(data)}. "
                "Ensure your input vector is the correct size."
            )

        return sum(data)  # pretend prediction
