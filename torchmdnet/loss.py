from torch.nn.functional import mse_loss, l1_loss, huber_loss

loss_class_mapping = {
    "mse_loss": mse_loss,
    "l1_loss": l1_loss,
    "huber_loss": huber_loss,
}
