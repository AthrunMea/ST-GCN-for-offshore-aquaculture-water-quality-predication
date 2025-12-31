import sys
from STGCN_MD.config import OptInit
from STGCN_MD.stgcn import STGCN
from STGCN_MD.MDJP import STGCN_MDJP
from utils.ckpt_util import save_checkpoint
import logging
from data_loader_STGCN.data_loaders import *
import math
from utils.util import metrix_seven
import warnings
warnings.filterwarnings("ignore")


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array([final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def main(logger):
    opt_class = OptInit(logger)
    opt = opt_class.get_args()
    logger.info('===> Creating dataloader ...')
    train_loader, valid_loader = data_generator_np(opt.data_dir, opt.task, opt.batch_size, device=opt.device)

    logger.info('===> Loading the network ...')
    trunk = STGCN(opt.water_para_num,opt.pred_para_num,1,opt.n_filters,opt.spatial_filters,
                  opt.input_step,opt.pred_step,opt.adj_mat,opt.device).to(opt.device)
    model = STGCN_MDJP(trunk=trunk, pred_step=48, n_filters=opt.n_filters).to(opt.device)
    logger.info('===> Init the optimizer ...')
    criterion = torch.nn.MSELoss(reduction='mean').to(opt.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=opt.lr_decay_rate,
                                                           patience=opt.optim_patience, eps=1e-08)
    logging.info('===> Start training ...')
    opt.best_mae = 1e10
    opt.best_score = []
    for e in range(opt.total_epochs):
        opt.epoch += 1
        train(model, train_loader, valid_loader, optimizer, scheduler, criterion, opt, logger)
        print('Best MAE:===>', opt.best_mae)
    if not os.path.exists('./result'):
        os.mkdir('./result')
    result_file = './result/{}_{}_{}.txt'.format(opt.model_name,str(opt.input_step),str(opt.pred_step))
    with open(result_file, "w") as file:
        for i in range(len(opt.best_score)):
            file.write(opt.best_score[i]+'\n')
    file.close()
    logger.info('Saving the final model.Finish!')


def train(model, train_loader, valid_data_loader, optimizer, scheduler, criterion, opt, logger):
    opt.train_loss = 0.0
    opt.train_mse   = [0]*opt.pred_para_num
    opt.train_mae = [0]*opt.pred_para_num
    opt.train_mape = [0]*opt.pred_para_num
    opt.train_rmse = [0]*opt.pred_para_num
    opt.train_rmspe = [0]*opt.pred_para_num
    opt.train_r2 = [0]*opt.pred_para_num
    opt.train_r = [0]*opt.pred_para_num
    opt.iter       = 0.
    model.train()
    for i, (data, target) in enumerate(train_loader):
        opt.iter += 1
        optimizer.zero_grad()
        pred = model(data)
        pred = torch.squeeze(pred)
        target = torch.squeeze(target.permute(0, 2, 1, 3))
        loss = criterion(pred, target)

        if opt.device==torch.device('cuda'):
            mse,mae,mape,rmse,rmspe,r2,r=metrix_seven(pred.detach().cpu().numpy(),target.cpu().numpy(),opt)
        else:
            mse,mae,mape,rmse,rmspe,r2,r=metrix_seven(pred.detach().numpy(),target.numpy(),opt)
        opt.train_mse  = [opt.train_mse[j]+mse[j]     for j in range(len(mse))]
        opt.train_mae=[opt.train_mae[j]+mae[j] for j in range(len(mae))]
        opt.train_mape=[opt.train_mape[j]+mape[j] for j in range(len(mape))]
        opt.train_rmse=[opt.train_rmse[j]+rmse[j] for j in range(len(rmse))]
        opt.train_rmspe=[opt.train_rmspe[j]+rmspe[j] for j in range(len(rmspe))]
        opt.train_r2=[opt.train_r2[j]+r2[j] for j in range(len(r2))]
        opt.train_r=[opt.train_r[j]+r[j] for j in range(len(r))]
        opt.train_loss += loss

        loss.backward()
        optimizer.step()
    scheduler.step(opt.train_loss)

    opt.train_loss  = opt.train_loss/opt.iter
    opt.train_mse   = [opt.train_mse[j]/opt.iter   for j in range(len(opt.train_mse))]
    opt.train_mae = [opt.train_mae[j]/opt.iter for j in range(len(opt.train_mae))]
    opt.train_mape = [opt.train_mape[j]/opt.iter for j in range(len(opt.train_mape))]
    opt.train_rmse = [opt.train_rmse[j]/opt.iter for j in range(len(opt.train_rmse))]
    opt.train_rmspe = [opt.train_rmspe[j]/opt.iter for j in range(len(opt.train_rmspe))]
    opt.train_r2  = [opt.train_r2[j]/opt.iter for j in range(len(opt.train_r2))]
    opt.train_r = [opt.train_r[j]/opt.iter for j in range(len(opt.train_r))]

    logger.info('Epoch:{}\t Epoch Train Loss: {:.4f}\t Mean MSE: {:.4f}\t Mean MAE: {:.4f}\t  '
                'Mean MAPE: {:.4f}\t Mean RMSE: {:.4f}\t'
                'Mean RMSPE: {:.4f}\t Mean R2: {:.4f}\t Mean R: {:.4f}\t Current_lr: {:.4f}'
                .format(opt.epoch, opt.train_loss, np.mean(opt.train_mse), np.mean(opt.train_mae),
                        np.mean(opt.train_mape),
                        np.mean(opt.train_rmse), np.mean(opt.train_rmspe), np.mean(opt.train_r2),
                        np.mean(opt.train_r), optimizer.param_groups[0]['lr']))
    for i in range(len(opt.para_name)):
        print('Epoch :{}\t {}===>MSE:{:.4f}\t MAE:{:.4f}\t MAPE:{:.4f}\t RMSE:{:.4f}\t '
              'RMSPE:{:.4f}\t R2:{:.4f}\t R:{:.4f}\t '
              .format(opt.epoch, opt.para_name[i], opt.train_mse[i], opt.train_mae[i], opt.train_mape[i],
                      opt.train_rmse[i], opt.train_rmspe[i], opt.train_r2[i], opt.train_r[i]))

    test(model, valid_data_loader, criterion, opt)
    opt.writer.add_scalars("Loss", {"Train": opt.train_loss}, opt.epoch)
    opt.writer.add_scalars("Loss", {"Validation": opt.vali_loss}, opt.epoch)
    if opt.epoch_vali_mae <= opt.best_mae:
        opt.best_mae = opt.epoch_vali_mae
        opt.best_score = ['Epoch Validation Loss: {:.4f}\t Mean MSE {:.4f}\t Mean MAE: {:.4f}\t  Mean MAPE: {:.4f}\t '
                'Mean RMSE: {:.4f}\t Mean RMSPE: {:.4f}\t Mean R2: {:.4f}\t Mean R: {:.4f}\t'
                .format(opt.vali_loss, np.mean(opt.vali_mse), np.mean(opt.vali_mae), np.mean(opt.vali_mape),
                        np.mean(opt.vali_rmse), np.mean(opt.vali_rmspe), np.mean(opt.vali_r2),
                        np.mean(opt.vali_r))]
        for i in range(len(opt.para_name)):
            opt.best_score.append('{}===>MSE:{:.4f}\t MAE:{:.4f}\t MAPE:{:.4f}\t RMSE:{:.4f}\t RMSPE:{:.4f}\t R2:{:.4f}\t R:{:.4f}\t '
                  .format(opt.para_name[i], opt.vali_mse[i], opt.vali_mae[i], opt.vali_mape[i],
                          opt.vali_rmse[i], opt.vali_rmspe[i], opt.vali_r2[i], opt.vali_r[i]))
        save_checkpoint({
            'epoch': opt.epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_value': opt.best_mae,
        }, opt.save_dir)


def test(model, test_loader, criterion, opt):
    opt.vali_mse   = [0]*opt.pred_para_num
    opt.vali_mae = [0]*opt.pred_para_num
    opt.vali_mape = [0]*opt.pred_para_num
    opt.vali_rmse = [0]*opt.pred_para_num
    opt.vali_rmspe = [0]*opt.pred_para_num
    opt.vali_r2 = [0]*opt.pred_para_num
    opt.vali_r = [0]*opt.pred_para_num
    opt.vali_loss = 0.0
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(opt.device), target.to(opt.device)
            target = torch.squeeze(target.permute(0, 2, 1, 3))
            pred = model(data)
            pred = torch.squeeze(pred)

            opt.vali_loss += criterion(pred, target)

            if opt.device == torch.device('cuda'):
                mse,mae, mape, rmse, rmspe, r2, r = metrix_seven(pred.detach().cpu().numpy(), target.cpu().numpy(), opt)
            else:
                mse,mae, mape, rmse, rmspe, r2, r = metrix_seven(pred.detach().numpy(), target.numpy(), opt)
            opt.vali_mse   = [opt.vali_mse[j]   + mse[j]   for j in range(len(mse))]
            opt.vali_mae = [opt.vali_mae[j] + mae[j] for j in range(len(mae))]
            opt.vali_mape = [opt.vali_mape[j] + mape[j] for j in range(len(mape))]
            opt.vali_rmse = [opt.vali_rmse[j] + rmse[j] for j in range(len(rmse))]
            opt.vali_rmspe = [opt.vali_rmspe[j] + rmspe[j] for j in range(len(rmspe))]
            opt.vali_r2 = [opt.vali_r2[j] + r2[j] for j in range(len(r2))]
            opt.vali_r = [opt.vali_r[j] + r[j] for j in range(len(r))]

    opt.vali_loss = opt.vali_loss / (i + 1)
    opt.vali_mse = [opt.vali_mse[j] / (i + 1) for j in range(len(opt.vali_mse))]
    opt.vali_mae = [opt.vali_mae[j] / (i + 1) for j in range(len(opt.vali_mae))]
    opt.vali_mape = [opt.vali_mape[j] / (i + 1) for j in range(len(opt.vali_mape))]
    opt.vali_rmse = [opt.vali_rmse[j] / (i + 1) for j in range(len(opt.vali_rmse))]
    opt.vali_rmspe = [opt.vali_rmspe[j] / (i + 1) for j in range(len(opt.vali_rmspe))]
    opt.vali_r2 = [opt.vali_r2[j] / (i + 1) for j in range(len(opt.vali_r2))]
    opt.vali_r = [opt.vali_r[j] / (i + 1) for j in range(len(opt.vali_r))]
    opt.epoch_vali_mae = np.mean(opt.vali_mae)
    logger.info('Epoch Validation Loss: {:.4f}\t Mean MSE {:.4f}\t Mean MAE: {:.4f}\t  Mean MAPE: {:.4f}\t '
                'Mean RMSE: {:.4f}\t Mean RMSPE: {:.4f}\t Mean R2: {:.4f}\t Mean R: {:.4f}\t'
                .format(opt.vali_loss, np.mean(opt.vali_mse), np.mean(opt.vali_mae), np.mean(opt.vali_mape),
                        np.mean(opt.vali_rmse), np.mean(opt.vali_rmspe), np.mean(opt.vali_r2),
                        np.mean(opt.vali_r)))
    for i in range(len(opt.para_name)):
        print('{}===>MSE:{:.4f}\t MAE:{:.4f}\t MAPE:{:.4f}\t RMSE:{:.4f}\t RMSPE:{:.4f}\t R2:{:.4f}\t R:{:.4f}\t '
              .format(opt.para_name[i], opt.vali_mse[i], opt.vali_mae[i], opt.vali_mape[i],
                      opt.vali_rmse[i], opt.vali_rmspe[i], opt.vali_r2[i], opt.vali_r[i]))


def make_logger():
    loglevel = "info"
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))
    log_format = logging.Formatter('%(asctime)s %(message)s')
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    logging.root = logger
    return logger


if __name__ == '__main__':
    logger = make_logger()
    main(logger)

