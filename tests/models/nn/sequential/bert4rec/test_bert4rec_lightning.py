import pytest

from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from replay.models.nn.optimizer_utils import FatLRSchedulerFactory, FatOptimizerFactory
    from replay.models.nn.sequential.bert4rec import Bert4Rec, Bert4RecPredictionDataset

torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")


@pytest.mark.torch
@pytest.mark.parametrize(
    "loss_type, loss_sample_count",
    [
        ("BCE", 6),
        ("CE", 6),
        ("BCE", None),
        ("CE", None),
    ],
)
def test_training_bert4rec_with_different_losses(
    item_user_sequential_dataset, train_loader, val_loader, loss_type, loss_sample_count
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type=loss_type,
        loss_sample_count=loss_sample_count,
    )
    trainer.fit(model, train_loader, val_loader)


@pytest.mark.torch
def test_init_bert4rec_with_invalid_loss_type(item_user_sequential_dataset):
    with pytest.raises(NotImplementedError) as exc:
        Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_type=""
        )

    assert str(exc.value) == "Not supported loss_type"


@pytest.mark.torch
def test_train_bert4rec_with_invalid_loss_type(item_user_sequential_dataset, train_loader):
    with pytest.raises(ValueError):
        trainer = L.Trainer(max_epochs=1)
        model = Bert4Rec(
            tensor_schema=item_user_sequential_dataset._tensor_schema,
            max_seq_len=5,
            hidden_size=64,
        )
        model._loss_type = ""
        trainer.fit(model, train_dataloaders=train_loader)


@pytest.mark.torch
def test_prediction_bert4rec(item_user_sequential_dataset, train_loader):
    pred = Bert4RecPredictionDataset(item_user_sequential_dataset, max_sequence_length=5)
    pred_loader = torch.utils.data.DataLoader(pred)
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
    )
    trainer.fit(model, train_loader)
    predicted = trainer.predict(model, pred_loader)

    assert len(predicted) == len(pred)
    assert predicted[0].size() == (1, 6)


@pytest.mark.torch
@pytest.mark.parametrize(
    "optimizer_factory, lr_scheduler_factory, optimizer_type, lr_scheduler_type",
    [
        (None, None, torch.optim.Adam, None),
        (FatOptimizerFactory(), None, torch.optim.Adam, None),
        (None, FatLRSchedulerFactory(), torch.optim.Adam, torch.optim.lr_scheduler.StepLR),
        (FatOptimizerFactory("sgd"), None, torch.optim.SGD, None),
        (FatOptimizerFactory(), FatLRSchedulerFactory(), torch.optim.Adam, torch.optim.lr_scheduler.StepLR),
    ],
)
def test_bert4rec_configure_optimizers(
    item_user_sequential_dataset,
    optimizer_factory,
    lr_scheduler_factory,
    optimizer_type,
    lr_scheduler_type,
):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        lr_scheduler_factory=lr_scheduler_factory,
        optimizer_factory=optimizer_factory,
    )

    parameters = model.configure_optimizers()
    if isinstance(parameters, tuple):
        assert isinstance(parameters[0][0], optimizer_type)
        assert isinstance(parameters[1][0], lr_scheduler_type)
    else:
        assert isinstance(parameters, optimizer_type)


@pytest.mark.torch
def test_bert4rec_configure_wrong_optimizer(item_user_sequential_dataset):
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        optimizer_factory=FatOptimizerFactory(""),
    )

    with pytest.raises(ValueError) as exc:
        model.configure_optimizers()

    assert str(exc.value) == "Unexpected optimizer"


@pytest.mark.torch
@pytest.mark.parametrize(
    "negative_sampling_strategy, negatives_sharing",
    [
        ("global_uniform", False),
        ("global_uniform", True),
        ("inbatch", False),
        ("inbatch", True),
    ],
)
def test_different_sampling_strategies(
    item_user_sequential_dataset, train_loader, val_loader, negative_sampling_strategy, negatives_sharing
):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema,
        max_seq_len=5,
        hidden_size=64,
        loss_type="BCE",
        loss_sample_count=6,
        negative_sampling_strategy=negative_sampling_strategy,
        negatives_sharing=negatives_sharing,
    )
    trainer.fit(model, train_loader, val_loader)


@pytest.mark.torch
def test_not_implemented_sampling_strategy(item_user_sequential_dataset, train_loader, val_loader):
    trainer = L.Trainer(max_epochs=1)
    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )
    model._negative_sampling_strategy = ""
    with pytest.raises(NotImplementedError):
        trainer.fit(model, train_loader, val_loader)


@pytest.mark.torch
def test_model_predict_with_nn_parallel(item_user_sequential_dataset, simple_masks):
    item_sequences, padding_mask, tokens_mask, _ = simple_masks

    model = Bert4Rec(
        tensor_schema=item_user_sequential_dataset._tensor_schema, max_seq_len=5, hidden_size=64, loss_sample_count=6
    )

    model._model = torch.nn.DataParallel(model._model)
    model._model_predict({"item_id": item_sequences}, padding_mask, tokens_mask)