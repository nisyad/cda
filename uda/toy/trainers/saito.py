import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import helper




def train_saito(model):  # sourcery skip: hoist-statement-from-loop

    G = Feature()  # Feature Generator
    C1 = Predictor()  # Classifier
    C2 = Predictor()  # Critic

    G.train()
    C1.train()
    C2.train()

    G_optimizer = None
    C1_optimizer = None
    C2_optimizer = None

    criterion = nn.CrossEntropyLoss()

    for batch_idx, ((imgs_src, lbls_src),
                    (imgs_tgt, _)) in enumerate(zip(trainloader_m,
                                                    trainloader_mm),
                                                start=1):

        imgs_src, lbls_src = imgs_src.to(device), lbls_src.to(device)

        reset_grad(G_optimizer, C1_optimizer, C2_optimizer)

        features_src = G(imgs_src)
        output_src = C1(features_src)

        loss_src = criterion(output_src, lbls_src)

        loss_src.backward()
        G_optimizer.step()
        C1_optimizer.step()

        reset_grad(G_optimizer, C1_optimizer, C2_optimizer)

        features_src = G(imgs_src)
        output_src = C2(features_src)

        loss_src = criterion(output_src, lbls_src)

        loss_src.backward()
        G_optimizer.step()
        C2_optimizer.step()

        reset_grad(G_optimizer, C1_optimizer, C2_optimizer)

        features_src = G(imgs_src)
        output_src = C1(features_src3)

        features_tgt = G(imgs_tgt)
        output_tgt1 = F.softmax(C1(features_tgt))
        output_tgt2 = F.softmax(C1(features_tgt))

        loss_src = criterion(output_src, lbls_src)
        loss_dis = discrepancy(output_tgt1, output_tgt2)
        loss = loss_src - loss_dis
        loss.backward()
        C1_optimizer.step()

        reset_grad(G_optimizer, C1_optimizer, C2_optimizer)

        for _ in range(4):
            features_tgt = G(imgs_tgt)
            output_tgt1 = F.softmax(C1(features_tgt))
            output_tgt2 = F.softmax(C1(features_tgt))

            loss_dis = discrepancy(output_tgt1, output_tgt2)
            loss_dis.backward()
            G_optimizer.step()
            reset_grad(G_optimizer, C1_optimizer, C2_optimizer)

        features_src = G(imgs_src)
        output_src1 = F.softmax(C1(features_src))
        output_src2 = F.softmax(C1(features_src))

        features_tgt = G(imgs_src)
        output_tgt1 = F.softmax(C1(features_tgt))
        output_tgt2 = F.softmax(C1(features_tgt))

        loss_dis = discrepancy(output_tgt1, output_tgt2)


def discrepancy(self, out1, out2):
    if self.entropy:
        return self.ent(out1)
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    if self.use_abs_diff:
        return torch.mean(torch.abs(out1 - out2))
    else:
        return (F.kl_div(F.log_softmax(out1), out2_t) +
                F.kl_div(F.log_softmax(out2), out1_t)) / 2


def reset_grad(G_optimizer, C1_optimizer, C2_optimizer):
    G_optimizer.zero_grad()
    C1_optimizer.zero_grad()
    C2_optimizer.zero_grad()


def ent(output):
    return -torch.mean(output * torch.log(output + 1e-6))
