import torch, time, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F 
from torch import optim
from tqdm import tqdm
import time 
from src.utils.metrics import Metrics
from src.utils.utils import cal_elapsed_time


# from apex import amp
# 混合精度加速

'''
目前实现的版本没法在shuffle了训练数据集的情况下算训练集Recall指标; 但训练集上的Recall指标不是很关键
'''
class Trainer(object):
    def __init__(self, config):
                
        # data
        if config.mode == 'run_all' or config.mode == 'run_train':
            self.trainset = config.dataset(**config.train_dataset_config)
            self.valset = config.dataset(**config.val_dataset_config)
            self.testset = config.dataset(**config.test_dataset_config)
            
        elif config.mode == 'run_val':
            self.valset = config.dataset(**config.val_dataset_config)
            
        elif config.mode == 'run_test' or config.mode == 'run_predict':
            self.testset = config.dataset(**config.test_dataset_config)

        self.model = config.model(config.model_config)
        
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        if config.use_cuda:
            config.logger.info('num of gpus : {}'.format(torch.cuda.device_count()))
            if config.use_multi_gpus:
                self.model = nn.DataParallel(self.model).to(self.device)
                config.logger.info('names of gpus : {}'.format(torch.cuda.get_device_name()))

            else:                
                self.model = self.model.cuda()
                config.logger.info('name of gpus : {}'.format(torch.cuda.get_device_name()))
            
        self.config = config
        
        if config.mode == 'run_all':
            self.writer = SummaryWriter(self.config.tensorboard_dir)

        self.criterion_loss = nn.BCELoss(reduction = 'mean') # mean example loss
        self.sigmoid_fn = nn.Sigmoid()
        self.criterion = lambda x, y : self.criterion_loss(self.sigmoid_fn(x), y.float())
        
        
        # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html#torch.nn.BCELoss


        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config.lr) 
        
        # self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, min_lr=1e-7,
        #                                                  patience=2, verbose=True, threshold=0.0001, eps=1e-8)
        
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.lr, momentum=0.9, dampening=0, weight_decay=self.config.l2_reg, nesterov=False)
        
        self.metrics = Metrics(config.score_file, config.num_cands)
        


        self.config.logger.info(self.model)


    def save_checkpoint(self, save_info, 
                        ckpt_file = None, 
                        ckpt_info_log_path = None
                        ):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        # torch.save(raw_model.state_dict(), save_path)
        
        # torch.save({'model_state_dict': raw_model.state_dict(), 
        #             'optimizer_state_dict': self.optimizer.state_dict(),
        #             }, save_path)
        
        
        # save disk space
        if ckpt_file is None:
            ckpt_file = self.config.ckpt_file
            
        torch.save({'model_state_dict': raw_model.state_dict(), 
                    }, ckpt_file)  

        self.config.logger.info('saved model into {} \n'.format(ckpt_file))
        
        # log
        if ckpt_info_log_path is None:
            ckpt_info_log_path = self.config.ckpt_info_log_path
            
        with open(ckpt_info_log_path,'w') as f:
            f.write(save_info + '\n')
            
            
    def load_checkpoint(self):
        ckpt = torch.load(self.config.ckpt_file)
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        raw_model.load_state_dict(ckpt['model_state_dict'])
        self.config.logger.info('loaded the trained model')

        
    def train(self, train_data_loader, val_data_loader):
        best_result = {'loss': None, 'R1': None}
        global_step = 0

        self.config.logger.info('Lr: {}'.format(self.optimizer.param_groups[0]['lr']))
        self.writer.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], global_step)
        
        for epoch in tqdm(range(self.config.num_epoch)):
            self.config.logger.info('>' * 100)
            self.config.logger.info('epoch: {}'.format(epoch + 1))
            n_step, n_sample_total, loss_total = 0, 0, 0 # in the epoch
            print_cnt = 0
            
            start_time = time.time()

            # switch model to training mode
            self.model.train()

            for batch_idx, batch_samples in enumerate(tqdm(train_data_loader)):
                global_step += 1
                n_step += 1
                
                # batchfy
                b_input_ids, b_masks, b_token_type_ids, b_label = batch_samples
                n_sample_total += b_input_ids.size(0) # 实际的example个数; 一轮对话的一个candidate为一个example
                
                ############# train  model #####################
                self.optimizer.zero_grad()
                
                logits = self.model(b_input_ids, b_masks, b_token_type_ids)
                 
                # logits: (b,)
                # b_label: (b,)
                
                loss = self.criterion(logits, b_label) # mean example loss 
                                
                loss.backward()
                
                if self.config.init_clip_max_norm is not None:                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], \
                            max_norm = self.config.init_clip_max_norm)
                    if grad_norm >= 1e2:
                        self.config.logger.info('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            
                self.optimizer.step()
                ####################################################
                
                loss_total += loss.item() 
                train_loss = loss_total / n_step # train_mean_example_loss in the epoch
                
                self.writer.add_scalar('Train/Loss', train_loss , global_step)

                if global_step % self.config.print_every == 0:
                    print_cnt += 1 
    
                    self.config.logger.info('epoch {}, iteration {}, '
                        'train_mean_example_loss: {:.4f}'
                        .format(epoch + 1, batch_idx + 1, train_loss))
                    
                # val
                if global_step % self.config.val_every == 0 or \
                        (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):

                    val_result = self.inference(val_data_loader, mode = 'val')
                    
                    self.writer.add_scalar('Val/Loss', val_result['loss'] , global_step) # mean_example_loss
                    self.writer.add_scalar('Val/Acc', val_result['acc'] , global_step)
                    self.writer.add_scalar('Val/R1', val_result['R1'] , global_step)
                    self.writer.add_scalar('Val/R2', val_result['R2'] , global_step)
                    self.writer.add_scalar('Val/R5', val_result['R5'] , global_step)
                    
                    
                    # remember
                    self.model.train()
                    
                    # adjust
                    if self.config.use_scheduler:
                        # self.scheduler.step()
                        self.scheduler.step(val_result['R1'])
                        self.config.logger.info('Lr: {}'.format(self.optimizer.param_groups[0]['lr']))
                        self.writer.add_scalar('Lr', self.optimizer.param_groups[0]['lr'], global_step)
                        
                        
                    
                    # save_best_only
                    
                    # if best_result['loss'] is None or resval_resultult['loss'] < best_result['loss']:
                    #     best_result = val_result
                    #     save_info = 'save info : epoch_{}_val_loss_{:.6f}_val_R1_{:.4f}'. \
                    #         format(epoch + 1, val_result['loss'], val_result['R1'])
                            
                    #     self.save_checkpoint(save_info)

                    if best_result['R1'] is None or val_result['R1'] > best_result['R1']:
                        best_result = val_result
                        save_info = 'save info : epoch_{}_val_loss_{:.6f}_val_R1_{:.4f}'. \
                            format(epoch + 1, val_result['loss'], val_result['R1'])
                            
                        self.save_checkpoint(save_info)
                        
                    
                    
                    elif (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):
                        save_info = 'save info : epoch_{}_val_loss_{:.6f}_val_R1_{:.4f}'. \
                            format(epoch + 1, val_result['loss'], val_result['R1'])
                            
                        num = global_step // self.config.force_save_every
                        
                        ckpt_info_log_path = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num,self.config.save_info_log_name
                        ))
                            
                        ckpt_force_file = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num, self.config.model_save_name
                        ))
                        self.config.logger.info('force save ...')
                        self.save_checkpoint(save_info, ckpt_force_file, ckpt_info_log_path)


            # # adjust
            # if self.config.use_scheduler:
            #     self.scheduler.step()
            #     # self.scheduler.step(r1)
                
            train_loss = loss_total / n_step # 一个epoch里的平均loss
            end_time = time.time()
            epoch_mins, epoch_secs = cal_elapsed_time(start_time, end_time)
            self.config.logger.info('epoch: {}, train_mean_sample_loss: {:.4f}, '
                'time: {}m {}s'. \
                    format(epoch + 1, train_loss, epoch_mins, epoch_secs))

    # b_input_ids: [bsz, num of seq, block_size]
    # def inference(self, data_loader, mode = 'val'):
    #     n_samples, loss_total = 0, 0
    #     n_batch = 0
    #     y_pred = [] # 1-d list
    #     y_label = [] # 1-d list
              
    #     self.model.eval()
        

    #     self.config.logger.info("{} ing ...".format(mode))
        
    #     with torch.no_grad():
    #         for batch_idx, batch_samples in enumerate(data_loader):
    #             n_batch += 1
                
    #             b_input_ids, b_attention_mask, b_token_type_ids, b_label = batch_samples

    #             # b_label [bsz, num of seq]
                
    #             the_real_batch_size = b_input_ids.size(0)
    #             n_samples += the_real_batch_size
                
                
    #             logits = self.model(b_input_ids, b_attention_mask, b_token_type_ids) # [bsz, num of seq]
                
                
    #             y_pred += logits.detach().cpu().numpy().tolist() # 2-d list
    #             y_label += b_label.detach().cpu().numpy().tolist() # 2-d list

                
    #             loss = self.criterion(logits, b_label)
        
    #             loss_total += loss.item() * the_real_batch_size

    #             if self.config.val_num is not None and batch_idx + 1 == self.config.val_num:
    #                 break
        
    #     # flatten
    #     y_pred = [logit for vec in y_pred for logit in vec]
    #     y_label = [label for vec in y_label for label in vec]
        
    #     pred_labels = [1 if score > 0 else 0 for score in y_pred] # 1-d list
    #     pred_result = [1 if pred_label == y_label[idx] else 0 for idx, pred_label in enumerate(pred_labels)]  
        
    #     acc = sum(pred_result) / len(pred_result)

    #     mean_sample_loss = loss_total / n_samples
        
        
    #     result_1 = {
    #         'acc' : acc,
    #         'loss' : mean_sample_loss
    #     }
        
    #     with open(self.config.score_file, 'w') as f:
    #         for score, label in zip(y_pred, y_label):
    #             f.write(
    #                 str(score) + '\t' +
    #                 str(label) + '\n'
    #             )
        
    #     R1,R2,R5 = self.metrics.evaluate_all_metrics()         
    #     result_2 = {
    #         'R1': R1,
    #         'R2': R2,
    #         'R5': R5
    #     }

    #     result = {**result_1, **result_2}

    #     self.config.logger.info("the whole/part of {} dataset:".format(mode))
    #     self.config.logger.info("n_samples : {}".format(n_samples)) # n_examples; Not n unique queries
    #     self.config.logger.info('loss: {:.4f}'.format(result['loss'])) # mean
    #     self.config.logger.info('acc : {}'.format(result['acc']))
    #     self.config.logger.info('R1: {:.4f}'.format(result['R1']))
    #     self.config.logger.info('R2: {:.4f}'.format(result['R2']))
    #     self.config.logger.info('R5: {:.4f}'.format(result['R5']))
    #     return result


    def inference(self, data_loader, mode = 'val'):
        n_samples, loss_total = 0, 0
        n_batch = 0
        y_pred = [] # 1-d list
        y_label = [] # 1-d list
              
        self.model.eval()
        

        self.config.logger.info("{} ing ...".format(mode))
        
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(data_loader)):
                n_batch += 1
                
                b_input_ids, b_attention_mask, b_token_type_ids, b_label = batch_samples

                # b_label: [bsz,]
                
                the_real_batch_size = b_input_ids.size(0)
                n_samples += the_real_batch_size
                
                
                logits = self.model(b_input_ids, b_attention_mask, b_token_type_ids) # [bsz,]
                
                y_pred += logits.cpu().detach().numpy().tolist() # 1-d list
                y_label += b_label.cpu().detach().numpy().tolist() # 1-d list

                            
                loss = self.criterion(logits, b_label)
        
                loss_total += loss.item() * the_real_batch_size

                if self.config.val_num is not None and batch_idx + 1 == self.config.val_num:
                    break
        
        pred_labels = [1 if score > 0 else 0 for score in y_pred] # 1-d list
        pred_result = [1 if pred_label == y_label[idx] else 0 for idx, pred_label in enumerate(pred_labels)]  
        
        acc = sum(pred_result) / len(pred_result)

        mean_sample_loss = loss_total / n_samples
        
        
        result_1 = {
            'acc' : acc,
            'loss' : mean_sample_loss
        }
        
        with open(self.config.score_file, 'w') as f:
            for score, label in zip(y_pred, y_label):
                f.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )
        
        R1,R2,R5 = self.metrics.evaluate_all_metrics()         
        result_2 = {
            'R1': R1,
            'R2': R2,
            'R5': R5
        }

        result = {**result_1, **result_2}

        self.config.logger.info("the whole/part of {} dataset:".format(mode))
        self.config.logger.info("n_samples : {}".format(n_samples)) # n_examples; Not n unique queries
        self.config.logger.info('loss: {:.4f}'.format(result['loss'])) # mean
        self.config.logger.info('acc : {}'.format(result['acc']))
        self.config.logger.info('R1: {:.4f}'.format(result['R1']))
        self.config.logger.info('R2: {:.4f}'.format(result['R2']))
        self.config.logger.info('R5: {:.4f}'.format(result['R5']))
        return result



    def predict_test(self, data_loader):
        # todo
        pass 
    
    def eval_test(self):
        pass 
    
          
    def run(self,mode = 'run_all'):
        self.config.logger.info("mode : {}".format(mode))
        
        if mode == 'run_all': 
            need_drop = False
            if self.config.use_cuda and self.config.use_multi_gpus:
                need_drop = True

            if self.config.sample_train_data:
                raise NotImplementedError
            
            else:
                shuffle = False
                if self.config.shuffle_on_the_fly:
                    shuffle = True
                    self.config.logger.info("shuffle_on_the_fly")
                
                train_data_loader = DataLoader(dataset = self.trainset, batch_size = self.config.batch_size, \
                    shuffle = shuffle, collate_fn=self.config.collect_fn, drop_last = need_drop) 
                # https://github.com/pytorch/pytorch/issues/42654
                # https://pytorch.org/docs/stable/data.html#working-with-collate-fn

            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                shuffle = False, collate_fn=self.config.collect_fn, drop_last = need_drop)
            
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                    shuffle = False, collate_fn=self.config.collect_fn, drop_last = need_drop)

            self.train(train_data_loader, val_data_loader)
            self.load_checkpoint()
            self.inference(test_data_loader, mode = 'test')
                        
        elif mode == 'run_val':
            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn=self.config.collect_fn)
            
            self.load_checkpoint()
            self.inference(val_data_loader, mode = 'val')
            
        elif mode == 'run_test':
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn=self.config.collect_fn)

            self.load_checkpoint()
            self.inference(test_data_loader, mode = 'test')
            # self.predict_test(test_data_loader)
                        
                        
        elif mode == 'run_predict':
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size, \
                    shuffle = False, collate_fn= self.config.collect_fn)
            
            self.load_checkpoint()
            # self.predict_test(test_data_loader)
            
            

