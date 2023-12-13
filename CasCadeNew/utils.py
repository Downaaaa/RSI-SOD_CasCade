def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr*decay
    # for param_group in optimizer:
    #     if param_group == 'lr':
    #        # print(1)
    #        param_group = init_lr*decay
    #        optimizer['lr'] = param_group
    #     else:
    #         continue
        # print('decay_epoch: {}, Current_LR: {}'.format(decay_epoch, param_group))