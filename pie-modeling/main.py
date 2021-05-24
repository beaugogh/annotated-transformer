if __name__ == '__main__':
    from classification_model import ClassificationModel, ClassificationArgs
    import pandas as pd
    import logging

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # Preparing train data
    train_data = [
        ["Aragorn was the heir of Isildur", 1.0],
        ["Frodo was the heir of Isildur", 0.0],
    ]
    train_df = pd.DataFrame(train_data)
    train_df.columns = ["text", "labels"]

    # Preparing eval data
    eval_data = [
        ["Theoden was the king of Rohan", 1.0],
        ["Merry was the king of Rohan", 0.0],
    ]
    eval_df = pd.DataFrame(eval_data)
    eval_df.columns = ["text", "labels"]

    # Optional model configuration
    model_args = ClassificationArgs(num_train_epochs=1)

    # Create a ClassificationModel
    # model = ClassificationModel(
    #     "roberta", "roberta-base", args=model_args
    # )

    model_path = '/home/bo/workspace/models/bert-base-uncased'
    model = ClassificationModel(
        "bert", model_path, args=model_args,  num_labels=1
    )

    # Train the model
    model.train_model(train_df)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_df)

    # Make predictions with the model
    predictions, raw_outputs = model.predict(["Sam was a Wizard"])

    print()