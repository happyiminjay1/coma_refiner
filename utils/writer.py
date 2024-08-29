import os
import time
import torch
import json
from glob import glob

from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, args=None):
        self.args = args

        if self.args is not None:
            tmp_log_list = glob(os.path.join(args.out_dir, 'log*'))
            if len(tmp_log_list) == 0:
                self.log_file = os.path.join(
                    args.out_dir, 'log_{:s}.txt'.format(
                        time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())))
            else:
                self.log_file = tmp_log_list[0]

    def print_info(self, info):
        message = 'Epoch: {}/{}, Duration: {:.3f}s, Train Loss: {:.4f}, Test Loss: {:.4f}, Train L1 Loss: {:.4f}, Train MANO L1 Loss: {:.4f}, Train Contact Loss: {:.4f}, Test L1 Loss: {:.4f}, Test MANO L1 Loss: {:.4f}, Train Contact Loss: {:.4f}, acc: {:.4f}, train_taxonomy_loss: {:.4f}, test_taxonomy_loss: {:.4f}, taxonomy_acc: {:.4f}' \
                .format(info['current_epoch'], info['epochs'], info['t_duration'], \
                info['train_loss'], info['test_loss'], info['train_l1'], info['train_mano_l1'], info['train_contact_loss'], info['test_l1'], info['test_mano_l1'], info['test_contact_loss'],info['acc'],info['train_taxonomy_loss'],info['test_taxonomy_loss'],info['taxonomy_acc'])
        with open(self.log_file, 'a') as log_file:
            log_file.write('{:s}\n'.format(message))
        print(message)

    def s_writer(self, info, s_writer, epoch):

        s_writer.add_scalar("Loss/train_loss", info['train_loss'], epoch)
        s_writer.add_scalar("Loss/test_loss", info['test_loss'], epoch)

        s_writer.add_scalar("Loss/train_l1", info['train_l1'], epoch)
        s_writer.add_scalar("Loss/train_mano_l1", info['train_mano_l1'], epoch)
        s_writer.add_scalar("Loss/train_contact_loss", info['train_contact_loss'], epoch)
        s_writer.add_scalar("Loss/train_taxonomy_loss", info['train_taxonomy_loss'], epoch)

        s_writer.add_scalar("Loss/test_l1", info['test_l1'], epoch)
        s_writer.add_scalar("Loss/test_mano_l1", info['test_mano_l1'], epoch)
        s_writer.add_scalar("Loss/test_contact_loss", info['test_contact_loss'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_loss", info['test_taxonomy_loss'], epoch)

        s_writer.add_scalar("Loss/test_contact_acc", info['acc'], epoch)
        s_writer.add_scalar("Loss/test_taxonomy_acc", info['taxonomy_acc'], epoch)

        s_writer.add_scalar("Loss/test_only_contact_acc", info['total_acc_c'], epoch)
        s_writer.add_scalar("Loss/test_only_non_contact_acc", info['total_acc_nc'], epoch)

        s_writer.add_scalar("Loss/precision", info['precision'], epoch)
        s_writer.add_scalar("Loss/recall", info['recall'], epoch)
        s_writer.add_scalar("Loss/f1_score", info['f1_score'], epoch)

        s_writer.flush()

    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            },
            os.path.join(self.args.checkpoints_dir,
                         'checkpoint_{:03d}.pt'.format(epoch)))
