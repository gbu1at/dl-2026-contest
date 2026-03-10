class Trainer():
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, criterion, device, cfg, logger) -> None:
        self.device = device
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.cfg = cfg
        self.logger = logger

    def train_epoch(self, i_epoch):
        self.model.train()
        for x, y in self.train_dataloader:
            x, y = x.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            self.logger.info(f"(train) epoch {i_epoch}: loss = {loss.item()}")

            loss.backward()
            self.optimizer.step()

    def validate(self):
        self.model.eval()
        for x, y in self.train_dataloader:
            x, y = x.to(self.device), y.to(self.device)
            y_pred = self.model(x)

            loss = self.criterion(y_pred, y)

            self.logger.info(f"(validate) : loss = {loss.item()}")
