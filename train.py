import os
import torch
from data import train_dataloader,valid_dataloader
from utils import Adder, Timer, check_lr
from torch.utils.tensorboard import SummaryWriter
from valid import _valid
import torch.nn.functional as F
from warmup_scheduler import GradualWarmupScheduler
from pytorch_msssim import ssim, ms_ssim
from accelerate import Accelerator
from accelerate.utils import set_seed

def _train(model, args):
    

    accelerator = Accelerator()
    set_seed(42)
    accelerator.print(f'device {str(accelerator.device)} is used!')

    model, optimizer, dataloader, ots = accelerator.prepare(
        model,
        torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8),
        train_dataloader(args.data_dir, args.batch_size, args.num_worker),
        valid_dataloader(args.data_dir, batch_size=1, num_workers=args.num_worker)
    )

    model.train()


    criterion = torch.nn.L1Loss()

    max_iter = len(dataloader)

    warmup_epochs=1
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch-warmup_epochs, eta_min=1e-5)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
    epoch = 1
    best_psnr=-1
    best_ep = -1 
    if args.resume:
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        best_ep = state['Bestep']
        best_psnr = state['Best']
        model.load_state_dict(state['model'])

        print('Resume from %d'%epoch)
        epoch += 1

    writer = SummaryWriter() if accelerator.is_local_main_process else None
    epoch_pixel_adder = Adder()
    epoch_fft_adder = Adder()
    
    iter_pixel_adder = Adder()
    iter_fft_adder = Adder()

    train_log = open(os.path.join(args.model_save_dir,'trainlog.txt'), mode = 'a',encoding='utf-8')

    for epoch_idx in range(epoch, args.num_epoch + 1):
        accelerator.wait_for_everyone()
        model.train()
        for iter_idx, batch_data in enumerate(dataloader):

            input_img, label_img = batch_data


            optimizer.zero_grad()
            
            pred_img = model(input_img)

            label_img2 = F.interpolate(label_img, scale_factor=0.5, mode='bilinear')
            label_img4 = F.interpolate(label_img, scale_factor=0.25, mode='bilinear')

            l1 = criterion(pred_img[0], label_img4)
            l2 = criterion(pred_img[1], label_img2)
            l3 = criterion(pred_img[2], label_img)

            loss_content = l1+l2+l3

            label_fft1 = torch.fft.fft2(label_img4, dim=(-2,-1))
            label_fft1 = torch.stack((label_fft1.real, label_fft1.imag), -1)

            pred_fft1 = torch.fft.fft2(pred_img[0], dim=(-2,-1))
            pred_fft1 = torch.stack((pred_fft1.real, pred_fft1.imag), -1)

            label_fft2 = torch.fft.fft2(label_img2, dim=(-2,-1))
            label_fft2 = torch.stack((label_fft2.real, label_fft2.imag), -1)

            pred_fft2 = torch.fft.fft2(pred_img[1], dim=(-2,-1))
            pred_fft2 = torch.stack((pred_fft2.real, pred_fft2.imag), -1)

            label_fft3 = torch.fft.fft2(label_img, dim=(-2,-1))
            label_fft3 = torch.stack((label_fft3.real, label_fft3.imag), -1)

            pred_fft3 = torch.fft.fft2(pred_img[2], dim=(-2,-1))
            pred_fft3 = torch.stack((pred_fft3.real, pred_fft3.imag), -1)

            f1 = criterion(pred_fft1, label_fft1)
            f2 = criterion(pred_fft2, label_fft2)
            f3 = criterion(pred_fft3, label_fft3)
            loss_fft = f1+f2+f3

            loss = loss_content + 0.1 * loss_fft 

            # loss.backward()
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
            optimizer.step()

            if accelerator.is_local_main_process:

                iter_pixel_adder(loss_content.item())
                iter_fft_adder(loss_fft.item())
                

                epoch_pixel_adder(loss_content.item())
                epoch_fft_adder(loss_fft.item())
                

                if (iter_idx + 1) % args.print_freq == 0:
                    print("Epoch: %03d Iter: %4d/%4d LR: %.10f Loss content: %7.4f  Loss fft: %7.4f" % (
                         epoch_idx, iter_idx + 1, max_iter, scheduler.get_lr()[0], iter_pixel_adder.average(),
                        iter_fft_adder.average()))
                    writer.add_scalar('Pixel Loss', iter_pixel_adder.average(), iter_idx + (epoch_idx-1)* max_iter)
                    writer.add_scalar('FFT Loss', iter_fft_adder.average(), iter_idx + (epoch_idx - 1) * max_iter)

                    iter_pixel_adder.reset()
                    iter_fft_adder.reset()

        
        scheduler.step()

        if accelerator.is_local_main_process:
            overwrite_name = os.path.join(args.model_save_dir, 'model.pkl')
            torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep},overwrite_name)


        if accelerator.is_local_main_process and epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep}, save_name)

            print("EPOCH: %02d\n Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f" % (
                epoch_idx, epoch_pixel_adder.average(), epoch_fft_adder.average()))
            train_log.write("EPOCH: %02d\n Epoch Pixel Loss: %7.4f Epoch FFT Loss: %7.4f \n" % (
                epoch_idx,  epoch_pixel_adder.average(), epoch_fft_adder.average()))
            epoch_fft_adder.reset()
            epoch_pixel_adder.reset()
            # epoch_ssim_adder.reset()

        
        if accelerator.is_local_main_process and epoch_idx % args.valid_freq == 0:
            
            # dist.barrier()
            model.eval()
            val = _valid(model.module, args, epoch_idx, ots)
            print('%03d epoch \n Average PSNR %.2f dB' % (epoch_idx, val))
            print('Best PSNR %.2f at %03d epoch ' % (best_psnr,best_ep))###############
            train_log.write('%03d epoch \n Average PSNR %.2f dB\n' % (epoch_idx, val))
            train_log.write('Best PSNR %.2f at %03d epoch \n' % (best_psnr,best_ep))
            
            writer.add_scalar('PSNR', val, epoch_idx)
            if val >= best_psnr:
                best_psnr = val ########
                best_ep = epoch_idx
                print('new-Best PSNR %.2f at %03d epoch ' % (best_psnr,best_ep))###############
                train_log.write('new-Best PSNR %.2f at %03d epoch \n' % (best_psnr,best_ep))
                torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep}, os.path.join(args.model_save_dir, 'Best.pkl'))
            writer.close()
            

        accelerator.wait_for_everyone()

    save_name = os.path.join(args.model_save_dir, 'Final.pkl')
    
    torch.save({'model': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch_idx,'Best':best_psnr,'Bestep':best_ep},save_name)
