import string
import argparse
import cv2

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import tensorflow as tf

from libs.DeepText.utils import CTCLabelConverter, AttnLabelConverter
from libs.DeepText.dataset import RawDataset, AlignCollate
from libs.DeepText.model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_Deeptext(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(num_class)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    return model, converter

def Deeptext_predict(img_path, opt, model, converter):
    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    image_folder = img_path
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    for image_tensors, image_path_list in demo_loader:
        print (image_tensors.size(0), image_path_list)
    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index, preds_size)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            print (1111111111111111111111111111)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}')
        return pred

if __name__ == '__main__':
    image_path = './demo_image/illusion.png'
    saved_model = './TPS-ResNet-BiLSTM-Attn.pth'
    model, Prediction, image_folder, converter = load_model_Deeptext(model_path, image_path)

    print(Deeptext_predict(saved_model, image_path, model, Prediction, image_folder, converter))
