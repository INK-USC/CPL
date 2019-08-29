e2_multi = ((1.0-Config.label_smoothing_epsilon)*e2_multi) + (1.0/e2_multi.size(1))

pred = model.forward(e1, rel)
loss = model.loss(pred, e2_multi)







***********************************
pred = model.forward(e1, rel)
loss = torch.zeros(1).cuda()
for j in range(128):
    position = torch.nonzero(e2_multi[j])[0].cuda()
    label = torch.cat([torch.ones(len(position)), torch.zeros(len(position))]).cuda()
    neg_position = torch.randint(e2_multi.shape[1], (len(position),)).long().cuda()
    position = torch.cat([position, neg_position])
    loss += model.loss(pred[j,position], label)