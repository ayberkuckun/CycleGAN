import wandb


def start(id=None):
    wandb.login()

    if id is None:
        wandb.init(project="colab-trials-cycleGAN", entity="turkish-hintons")
    else:
        wandb.init(project="colab-trials-cycleGAN", entity="turkish-hintons", resume=True, id=id)

    wandb.define_metric("gen_G_loss", summary="min")
    wandb.define_metric("gen_F_loss", summary="min")
    wandb.define_metric("cyclic_loss", summary="min")
    wandb.define_metric("identity_loss", summary="min")
    wandb.define_metric("disc_X_loss", summary="min")
    wandb.define_metric("disc_Y_loss", summary="min")
    wandb.define_metric("total_gen_G_loss", summary="min")
    wandb.define_metric("total_gen_F_loss", summary="min")
