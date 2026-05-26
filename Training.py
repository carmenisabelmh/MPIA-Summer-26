

def mse_loss(Y, Yhat, M):
        err = ((Y - Yhat) ** 2).sum(dim=-1)  # [B, T]  squared L2 norm over patch dim
        return err[M].mean() 