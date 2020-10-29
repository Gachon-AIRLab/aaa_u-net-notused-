import torchvision
from GilAAADataset import *
from engine import train_one_epoch, evaluate
import pickle
import cv2
import transforms as T
from torchvision.transforms import functional as F
from torchvision import transforms


from gil_eval import *
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def main(mode):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and ...
    num_classes = 2

    # get the model using our helper function
    model = UNet(n_channels=1, n_classes=1, bilinear=True)

    # move model to the right device
    model.to(device)

    if 'test' in mode:
        model.load_state_dict(torch.load('./checkpoints/CP_epoch10.pth'))

        # img, _ = dataset_test[13]
        # print(img.shape)
        check_dir = '../AAAGilDatasetPos/'
        subject = '05390853_20200821'
        img_idx = 96

        img_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/' + subject + '_%04d.png'%img_idx
        mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/' + subject + '_%04d.png'%img_idx
        #
        # img = cv2.imread('AAAGilDatasetPos/img/img_0001_0010.png', 1)
        # seg = cv2.imread('AAAGilDatasetPos/mask/mask_0001_001 0.png', 1)

        img = Image.open(img_name)
        mask_gt = Image.open(mask_name)

        img = torch.from_numpy(BasicDataset.preprocess(img, 1))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        # put the model in evaluation mode
        model.eval()
        with torch.no_grad():
            prediction = model(img.to(device))
            #prediction = prediction.squeeze(0)
            # probs = torch.sigmoid(prediction)
            # print(probs)
            #print(prediction.shape)

        #im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        ##mask = Image.fromarray(prediction[0, 0].mul(255).byte().cpu().numpy())

        probs = torch.sigmoid(prediction)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
        print(full_mask)

        mask = Image.fromarray(full_mask)


        
        mask.show()
        mask_gt.show()

        img_in = np.array(im)
        img_mask = np.array(mask)

        img_mask_gt = np.array(mask_gt)

        img_mask[img_mask > 50] = 255
        img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)

        # cv2.imshow('input', img_in)
        # cv2.imshow('mask result', img_mask)
        # cv2.imshow('mask gt', img_mask_gt)

        # segment evaluation
        overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt_gray)
        print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f'%(overlap, jaccard, dice, fn, fp))


        img_result = np.concatenate([img_mask, img_mask_gt_gray], axis=1)

        img_overlap = img_mask_gt.copy()
        img_overlap[:, :, 0] = 0
        img_overlap[:,:,1] = img_mask
        cv2.imshow('result', img_result)
        cv2.imshow('overlap', img_overlap)
        cv2.waitKey(0)




    if 'eval' in mode:
        model.load_state_dict(torch.load('./checkpoints/CP_epoch10.pth'))
        dataset_test = torch.load('./pretrained/dataset_test.pth')

        num_test = len(dataset_test.indices)
        mat_eval = np.zeros((num_test, 5), np.float32)


        for i in range(num_test):
            img_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]

            img = Image.open(img_name).convert("RGB")
            mask_gt = Image.open(mask_name).convert("RGB")
            img_rgb = np.array(img)
            img = F.to_tensor(img)

            model.eval()
            with torch.no_grad():
                prediction = model([img.to(device)])

            im = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())


            img_mask = np.array(mask)
            img_mask_gt = np.array(mask_gt)
            img_mask[img_mask > 50] = 255
            img_mask_gt_gray = cv2.cvtColor(img_mask_gt, cv2.COLOR_BGR2GRAY)

            overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt_gray)
            print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            overlap, jaccard, dice, fn, fp))

            mat_eval[i, :] = [overlap, jaccard, dice, fn, fp]


            img_gray  = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

            img_result = np.concatenate([img_gray, img_mask, img_mask_gt_gray], axis=1)


            img_overlap = img_mask_gt.copy()
            img_overlap[:, :, 0] = 0
            img_overlap[:, :, 1] = img_mask



            cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_raw.png'), img_gray)
            cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_mask_predict.png'), img_mask)
            cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_mask_gt.png'), img_mask_gt_gray)
            cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_overlap.png'), img_overlap)
            cv2.imwrite('result/' + dataset_test.dataset.imgs[dataset_test.indices[i]].replace('.png', '_all.png'), img_result)

            cv2.imshow('result', img_result)
            cv2.imshow('overlap', img_overlap)
            cv2.waitKey(10)

        print(mat_eval.mean(axis=0))



if __name__ == '__main__':
    main('test')