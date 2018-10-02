import torch


class Trainer(object):
    def __init__(self, xs, ys, model, criterion, optimizer):
        self.xs = xs
        self.ys = ys
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def _make_batch(cls, data, batch_size=100):
        N, D, T = data.shape
        num_batch = N // batch_size
        if N % batch_size != 0:
            num_batch += 1

        batches = []
        for i in range(num_batch):
            idx = i * batch_size
            batches.append(data[idx:idx+batch_size, :])
        return batches

    def train(self, max_epoch, batch_size=100):
        x_batches = self._make_batch(self.xs, batch_size=batch_size)
        y_batches = self._make_batch(self.ys, batch_size=batch_size)
        
        for epoch in range(max_epoch):
            running_loss = 0.0

            for x_batch, y_batch in zip(x_batches, y_batches):
                # torch.tensor形式への変換
                x = torch.tensor(x_batch, dtype=torch.float)
                y = torch.tensor(y_batch, dtype=torch.float)

                # lstmへの入力は (N, T, D) だが、現在は (N, D, T) なので
                # 軸を入れ替える
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)

                # optimizerの初期化
                self.optimizer.zero_grad()

                # モデルに入力を入れて誤差の計算
                output = self.model(x)
                loss = self.criterion(output, y)

                # 誤差を元に逆伝播を行い、パラメータを更新
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            if epoch % 20 == 0:
                print('epoch: %d | loss: %.3f' % (epoch, running_loss))
