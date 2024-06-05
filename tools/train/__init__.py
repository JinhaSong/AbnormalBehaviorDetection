class GCLModel(pl.LightningModule):
    def __init__(self,
                 path='../data/shanghai',
                 batch_size=16,
                 lr=1e-5,
                 lr_step=10,
                 worker_n=1,
                 clip_len=16,
                 self_pretrain=False,
                 anoamly_fit=False,
                 neg_learning=True,
                 coop=True):

        super(GCLModel, self).__init__()
        self.lr = lr
        self.lr_step = lr_step
        self.self_pretrain = self_pretrain
        self.anoamly_fit = self.anoamly_fit
        self.data_dir = path
        self.batch_size = batch_size
        self.num_workers = worker_n
        self.clip_len = clip_len

        # HeatmapC3D 모델 초기화
        self.feature_extractor = HeatmapC3D()
        feat_size = 256  # HeatmapC3D 출력 크기

        self.ae = Autoencoder(feat_size=feat_size)
        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()

        if coop:
            self.dis = Discriminator(feat_size=feat_size)

            self.forward = self.forward_coop
            self.video_predict = self.video_predict_coop

            if neg_learning:
                self.gen_dis_loss = self.gen_dis_loss_neg_learn
            else:
                self.gen_dis_loss = self.gen_dis_loss_no_neg

            self.val_auc_loss = self.val_auc_loss_coop

            self.training_step = self.training_step_coop
            self.validation_step = self.validation_step_coop
        else:
            self.forward = self.forward_ae
            self.video_predict = self.video_predict_ae
            self.training_step = self.training_step_ae
            self.validation_step = self.validation_step_ae

    def get_dis_thresh(self, dis_pred):
        d_std, d_mean = torch.std_mean(dis_pred, unbiased=True)
        return d_mean + 0.1 * d_std

    def get_gen_thresh(self, gen_pred):
        g_std, g_mean = torch.std_mean(gen_pred, unbiased=True)
        return g_std + g_mean

    def val_auc_loss_coop(self, gen_pred, dis_pred, label):
        g_thresh = self.get_gen_thresh(gen_pred)
        d_thresh = self.get_dis_thresh(dis_pred)

        gen_pred = (gen_pred > g_thresh).float()
        dis_pred = (dis_pred > d_thresh).float()

        gen_auc = auroc(gen_pred.cpu(), label.cpu(), num_classes=None)
        dis_auc = auroc(dis_pred.cpu(), label.cpu(), num_classes=None)
        return gen_auc, dis_auc

    def gen_dis_loss_neg_learn(self, x_hat, gen_y, dis_y):
        g_dist = self.mse_loss(gen_y, x_hat)
        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)

        dis_y_pseu = (batch_dist > g_thresh).float()
        d_loss = self.bce_loss(dis_y, dis_y_pseu)
        d_thresh = self.get_dis_thresh(dis_y)
        mask = dis_y < d_thresh
        gen_y[mask] = torch.ones_like(gen_y[mask])
        g_loss = self.mse_loss(gen_y, x_hat).mean()
        return g_loss, d_loss

    def gen_dis_loss_no_neg(self, x_hat, gen_y, dis_y):
        g_dist = self.mse_loss(gen_y, x_hat)
        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)

        dis_y_pseu = (batch_dist > g_thresh).float()
        d_loss = self.bce_loss(dis_y, dis_y_pseu)
        g_loss = self.mse_loss(gen_y, x_hat).mean()
        return g_loss, d_loss

    def forward_ae(self, x):
        x = self.feature_extractor(x)
        g_y = self.ae(x)
        return x, g_y

    def forward_coop(self, x):
        x = self.feature_extractor(x)
        g_y = self.ae(x)
        d_y = self.dis(x)
        return x, g_y, d_y

    def video_predict_ae(self, x):
        x_hat, gen_y = self(x)
        g_dist = self.mse_loss(gen_y, x_hat)
        g_preds = g_dist.mean(dim=1)
        return g_preds

    def video_predict_coop(self, x):
        x_hat, gen_y, dis_pred = self(x)
        g_dist = self.mse_loss(gen_y, x_hat)
        gen_pred = g_dist.mean(dim=1)
        return gen_pred, dis_pred

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr, alpha=0.6)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.lr_step)
        return [optimizer], [lr_scheduler]

    def training_step_ae(self, batch, batch_idx):
        x, label = batch
        x_hat, gen_y = self(x)
        g_dist = self.mse_loss(gen_y, x_hat)

        batch_dist = g_dist.mean(dim=1)
        g_thresh = self.get_gen_thresh(batch_dist)
        g_preds = (batch_dist > g_thresh).float()

        loss = batch_dist.mean()
        g_auc = auroc(g_preds, label, num_classes=None)

        self.log('train/g_auc', g_auc, prog_bar=True)
        self.log('train/loss', loss)
        return loss

    def training_step_coop(self, batch, batch_idx):
        x, label = batch
        x_hat, gen_y, dis_y = self(x)
        g_loss, d_loss = self.gen_dis_loss(x_hat, gen_y, dis_y)
        loss = g_loss + d_loss

        self.log('train/g_loss', g_loss, prog_bar=True)
        self.log('train/d_loss', d_loss, prog_bar=True)
        self.log('train/loss', loss)
        return loss

    def validation_step_ae(self, batch, batch_idx):
        x, label = batch
        batch_dist = self.video_predict(x)

        g_thresh = self.get_gen_thresh(batch_dist)
        g_preds = (batch_dist > g_thresh).float()
        loss = batch_dist.mean()
        g_auc = auroc(g_preds, label, num_classes=None)

        self.log("test/g_mse", loss, prog_bar=True)
        self.log("test/g_auc", g_auc, prog_bar=True)
        return g_auc

    def validation_step_coop(self, batch, batch_idx):
        x, label = batch
        gen_pred, dis_pred = self.video_predict(x)
        loss = gen_pred.mean()
        gen_auc, dis_auc = self.val_auc_loss(gen_pred, dis_pred, label)

        self.log("test/g_mse", loss, prog_bar=True)
        self.log("test/g_auc", gen_auc, prog_bar=True)
        self.log("test/d_auc", dis_auc, prog_bar=True)

        return dis_auc

    def get_feat_size(self):
        dataset = ClipDataset(self.data_dir)
        return len(dataset[0][0])

    def train_dataloader(self):
        dataset = ClipDataset(self.data_dir,
                              split='train',
                              clean=self.self_pretrain)
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

    def val_dataloader(self):
        dataset = ClipDataset(self.data_dir, split='test')
        return DataLoader(dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=True)