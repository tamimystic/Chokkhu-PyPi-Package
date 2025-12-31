class Trainer:
    def fit(self, model, train_ds, val_ds, epochs):
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
        )
        return history
