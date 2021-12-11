import numpy as np
from . import utils, dynamics
from numba import jit
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import convolve
import itertools, tabulate, logging, types, os, cv2


def mask_ious(masks_true, masks_pred):
    """ return best-matched masks """
    iou = _intersection_over_union(masks_true, masks_pred)[1:,1:]
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= 0.5).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    iout = np.zeros(masks_true.max())
    iout[true_ind] = iou[true_ind,pred_ind]
    preds = np.zeros(masks_true.max(), 'int')
    preds[true_ind] = pred_ind+1
    return iout, preds

def boundary_scores(masks_true, masks_pred, scales):
    """ boundary precision / recall / Fscore """
    diams = [utils.diameters(lbl)[0] for lbl in masks_true]
    precision = np.zeros((len(scales), len(masks_true)))
    recall = np.zeros((len(scales), len(masks_true)))
    fscore = np.zeros((len(scales), len(masks_true)))
    for j, scale in enumerate(scales):
        for n in range(len(masks_true)):
            diam = max(1, scale * diams[n])
            rs, ys, xs = utils.circleMask([int(np.ceil(diam)), int(np.ceil(diam))])
            filt = (rs <= diam).astype(np.float32)
            otrue = utils.masks_to_outlines(masks_true[n])
            otrue = convolve(otrue, filt)
            opred = utils.masks_to_outlines(masks_pred[n])
            opred = convolve(opred, filt)
            tp = np.logical_and(otrue==1, opred==1).sum()
            fp = np.logical_and(otrue==0, opred==1).sum()
            fn = np.logical_and(otrue==1, opred==0).sum()
            precision[j,n] = tp / (tp + fp)
            recall[j,n] = tp / (tp + fn)
        fscore[j] = 2 * precision[j] * recall[j] / (precision[j] + recall[j])
    return precision, recall, fscore


def aggregated_jaccard_index(masks_true, masks_pred):
    """ AJI = intersection of all matched masks / union of all masks 
    
    Parameters
    ------------

    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    aji : aggregated jaccard index for each set of masks

    """

    aji = np.zeros(len(masks_true))
    for n in range(len(masks_true)):  # 对于每张图片
        iout, preds = mask_ious(masks_true[n], masks_pred[n])  # 返回best match
        inds = np.arange(0, masks_true[n].max(), 1, int)  # 最大为实例个数（默认=最大实例值）
        overlap = _label_overlap(masks_true[n], masks_pred[n])  # 像素交，实例对应的值不一样呢？这里有个大前提
        union = np.logical_or(masks_true[n]>0, masks_pred[n]>0).sum()  # 实例上的并
        overlap = overlap[inds[preds>0]+1, preds[preds>0].astype(int)]
        aji[n] = overlap.sum() / union
    return aji 


def average_precision(masks_true, masks_pred, threshold=[0.5, 0.75, 0.9], return_pred_iou=False):
    """ average precision estimation: AP = TP / (TP + FP + FN)

    This function is based heavily on the *fast* stardist matching functions
    (https://github.com/mpicbg-csbd/stardist/blob/master/stardist/matching.py)

    Parameters
    ------------
    
    masks_true: list of ND-arrays (int) or ND-array (int) 
        where 0=NO masks; 1,2... are mask labels
    masks_pred: list of ND-arrays (int) or ND-array (int) 
        ND-array (int) where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    ap: array [len(masks_true) x len(threshold)]
        average precision at thresholds
    tp: array [len(masks_true) x len(threshold)]
        number of true positives at thresholds
    fp: array [len(masks_true) x len(threshold)]
        number of false positives at thresholds
    fn: array [len(masks_true) x len(threshold)]
        number of false negatives at thresholds

    """
    not_list = False
    if not isinstance(masks_true, list):
        masks_true = [masks_true]
        masks_pred = [masks_pred]
        not_list = True
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    ap  = np.zeros((len(masks_true), len(threshold)), np.float32)
    tp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fp  = np.zeros((len(masks_true), len(threshold)), np.float32)
    fn  = np.zeros((len(masks_true), len(threshold)), np.float32)
    if return_pred_iou: pred_ious  = []
    # 这里有bug，序号最大不一定代表个数
    # n_true = np.array(list(map(np.max, masks_true)))
    # n_pred = np.array(list(map(np.max, masks_pred)))
    # 现在改成这样
    n_true = np.array([len(set(x.flatten()))-1 for x in masks_true])
    n_pred = np.array([len(set(x.flatten()))-1 for x in masks_pred])

    iou = None
    for n in range(len(masks_true)):
        #_,mt = np.reshape(np.unique(masks_true[n], return_index=True), masks_pred[n].shape)
        if n_pred[n] > 0:
            iou = _intersection_over_union(masks_true[n], masks_pred[n])[1:, 1:]
            for k,th in enumerate(threshold):
                tp[n,k] = _true_positive(iou, th)
        fp[n] = n_pred[n] - tp[n]
        fn[n] = n_true[n] - tp[n]
        ap[n] = tp[n] / (tp[n] + fp[n] + fn[n])
        if return_pred_iou: pred_ious.append(np.array(np.max(iou, axis=0)))

    if not_list:
        ap, tp, fp, fn = ap[0], tp[0], fp[0], fn[0]
    if return_pred_iou:
        return ap, tp, fp, fn, pred_ious
    return ap, tp, fp, fn

@jit(nopython=True)
def _label_overlap(x, y):
    """ fast function to get pixel overlaps between masks in x and y 
    
    Parameters
    ------------

    x: ND-array, int
        where 0=NO masks; 1,2... are mask labels
    y: ND-array, int
        where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    overlap: ND-array, int
        matrix of pixel overlaps of size [x.max()+1, y.max()+1]
    
    """
    x = x.ravel()
    y = y.ravel()
    overlap = np.zeros((1+x.max(),1+y.max()), dtype=np.uint)
    for i in range(len(x)):
        overlap[x[i],y[i]] += 1
    return overlap

def _intersection_over_union(masks_true, masks_pred):
    """ intersection over union of all mask pairs
    
    Parameters
    ------------
    
    masks_true: ND-array, int 
        ground truth masks, where 0=NO masks; 1,2... are mask labels
    masks_pred: ND-array, int
        predicted masks, where 0=NO masks; 1,2... are mask labels

    Returns
    ------------

    iou: ND-array, float
        matrix of IOU pairs of size [x.max()+1, y.max()+1]

    """
    overlap = _label_overlap(masks_true, masks_pred)
    n_pixels_pred = np.sum(overlap, axis=0, keepdims=True)
    n_pixels_true = np.sum(overlap, axis=1, keepdims=True)
    iou = overlap / (n_pixels_pred + n_pixels_true - overlap)
    iou[np.isnan(iou)] = 0.0
    return iou

def _true_positive(iou, th):
    """ true positive at threshold th
    
    Parameters
    ------------

    iou: float, ND-array
        array of IOU pairs
    th: float
        threshold on IOU for positive label

    Returns
    ------------

    tp: float
        number of true positives at threshold

    """
    n_min = min(iou.shape[0], iou.shape[1])
    costs = -(iou >= th).astype(float) - iou / (2*n_min)
    true_ind, pred_ind = linear_sum_assignment(costs)
    match_ok = iou[true_ind, pred_ind] >= th
    tp = match_ok.sum()
    return tp

def flow_error(maski, dP_net):
    """ error in flows from predicted masks vs flows predicted by network run on image

    This function serves to benchmark the quality of masks, it works as follows
    1. The predicted masks are used to create a flow diagram
    2. The mask-flows are compared to the flows that the network predicted

    If there is a discrepancy between the flows, it suggests that the mask is incorrect.
    Masks with flow_errors greater than 0.4 are discarded by default. Setting can be
    changed in Cellpose.eval or CellposeModel.eval.

    Parameters
    ------------
    
    maski: ND-array (int) 
        masks produced from running dynamics on dP_net, 
        where 0=NO masks; 1,2... are mask labels
    dP_net: ND-array (float) 
        ND flows where dP_net.shape[1:] = maski.shape

    Returns
    ------------

    flow_errors: float array with length maski.max()
        mean squared error between predicted flows and flows from masks
    dP_masks: ND-array (float)
        ND flows produced from the predicted masks
    
    """
    if dP_net.shape[1:] != maski.shape:
        print('ERROR: net flow is not same size as predicted masks')
        return
    maski = np.reshape(np.unique(maski.astype(np.float32), return_inverse=True)[1], maski.shape)
    # flows predicted from estimated masks
    dP_masks,_ = dynamics.masks_to_flows(maski)  # 他会根据预测出来的mask重新生成flow来对原flow进行质量控制
    iun = np.unique(maski)[1:]
    flow_errors=np.zeros((len(iun),))
    for i,iu in enumerate(iun):
        ii = maski==iu
        if dP_masks.shape[0]==2:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2).mean()
        else:
            flow_errors[i] += ((dP_masks[0][ii] - dP_net[0][ii]/5.)**2 * 0.5
                            + (dP_masks[1][ii] - dP_net[1][ii]/5.)**2
                            + (dP_masks[2][ii] - dP_net[2][ii]/5.)**2).mean()
    return flow_errors, dP_masks


def panoptic_quality(masks_true, masks_pred, threshold=[0.5]):
    """`threshold` is the IoU threshold level to determine the pairing between
    GT instances `p` and prediction instances `g`. `p` and `g` is a pair
    if IoU > `match_iou`. However, pair of `p` and `g` must be unique
    (1 prediction instance to 1 GT instance mapping).

    If `match_iou` < 0.5, Munkres assignment (solving minimum weight matching
    in bipartite graphs) is caculated to find the maximal amount of unique pairing.

    If `match_iou` >= 0.5, all IoU(p,g) > 0.5 pairing is proven to be unique and
    the number of pairs is also maximal.

    Fast computation requires instance IDs are in contiguous orderding
    i.e [1, 2, 3, 4] not [2, 3, 6, 10]. Please call `remap_label` beforehand
    and `by_size` flag has no effect on the result.

    Returns:
        [dq, sq, pq]: measurement statistic

        [paired_true, paired_pred, unpaired_true, unpaired_pred]:
                      pairing information to perform measurement

    """
    if not isinstance(threshold, list) and not isinstance(threshold, np.ndarray):
        threshold = [threshold]
    dq  = np.zeros((len(masks_true), len(threshold)), np.float32)
    sq  = np.zeros((len(masks_true), len(threshold)), np.float32)
    pq  = np.zeros((len(masks_true), len(threshold)), np.float32)

    for n in range(len(masks_true)):
        masks_truen = np.copy(masks_true[n])
        masks_predn = np.copy(masks_pred[n])
        true_id_list = list(np.unique(masks_truen))
        pred_id_list = list(np.unique(masks_predn))

        # prefill with value
        pairwise_iou = np.zeros(
            [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
        )  # 去掉0

        # caching pairwise iou

        for true_id in true_id_list[1:]:  # 0-th is background
            t_mask = np.array(masks_truen == true_id, np.uint8)
            pred_true_overlap = masks_predn[t_mask > 0]  # 真实mask的区域在预测中抠出来
            pred_true_overlap_id = np.unique(pred_true_overlap)  #
            pred_true_overlap_id = list(pred_true_overlap_id)
            for pred_id in pred_true_overlap_id:
                if pred_id == 0:  # ignore
                    continue  # overlaping background
                p_mask =  np.array(masks_predn == pred_id, np.uint8)
                total = (t_mask + p_mask).sum()
                inter = (t_mask * p_mask).sum()
                iou = inter / (total - inter)
                pairwise_iou[true_id-1, pred_id-1] = iou
        for k, th in enumerate(threshold):
            assert th >= 0.0, "Cant' be negative"
            if th >= 0.5:
                paired_iou = pairwise_iou[pairwise_iou > th]
                pairwise_iou[pairwise_iou <= th] = 0.0
                paired_true, paired_pred = np.nonzero(pairwise_iou)
                paired_iou = pairwise_iou[paired_true, paired_pred]
                paired_true += 1  # index is instance id - 1
                paired_pred += 1  # hence return back to original
            else:  # * Exhaustive maximal unique pairing
                #### Munkres pairing with scipy library
                # the algorithm return (row indices, matched column indices)
                # if there is multiple same cost in a row, index of first occurence
                # is return, thus the unique pairing is ensure
                # inverse pair to get high IoU as minimum
                paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
                ### extract the paired cost and remove invalid pair
                paired_iou = pairwise_iou[paired_true, paired_pred]

                # now select those above threshold level
                # paired with iou = 0.0 i.e no intersection => FP or FN
                paired_true = list(paired_true[paired_iou > th] + 1)
                paired_pred = list(paired_pred[paired_iou > th] + 1)
                paired_iou = paired_iou[paired_iou > th]

            # get the actual FP and FN
            unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
            unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]
            # print(paired_iou.shape, paired_true.shape, len(unpaired_true), len(unpaired_pred))

            tp = len(paired_true)
            fp = len(unpaired_pred)
            fn = len(unpaired_true)
            # get the F1-score i.e DQ
            dq[n, k] = tp / (tp + 0.5 * fp + 0.5 * fn)
            # get the SQ, no paired has 0 iou so not impact
            sq[n, k] = paired_iou.sum() / (tp + 1.0e-6)
            pq[n,k] = dq[n, k] * sq[n, k]

    return dq, sq, pq


def type_stat(classes_true, classes_pred, masks_true, masks_pred, type_uid_list=None, exhaustive=True):
    """GT must be exhaustively annotated for instance location (detection).

    Args:
        true_dir, pred_dir: Directory contains .mat annotation for each image.
                            Each .mat must contain:
                    --`inst_centroid`: Nx2, contains N instance centroid
                                       of mass coordinates (X, Y)
                    --`inst_type`    : Nx1: type of each instance at each index
                    `inst_centroid` and `inst_type` must be aligned and each
                    index must be associated to the same instance
        type_uid_list : list of id for nuclei type which the score should be calculated.
                        Default to `None` means available nuclei type in GT.
        exhaustive : Flag to indicate whether GT is exhaustively labelled
                     for instance types

    """
    paired_all = []  # unique matched index pair
    unpaired_true_all = []  # the index must exist in `true_inst_type_all` and unique
    unpaired_pred_all = []  # the index must exist in `pred_inst_type_all` and unique
    true_inst_type_all = []  # each index is 1 independent data point
    pred_inst_type_all = []  # each index is 1 independent data point
    for i in range(len(classes_pred)):  # 每次读取一张图像
        true_inst_mask = masks_true[i].astype("int32")
        true_inst_type_img = classes_true[i].astype("int32")
        true_centroid, true_inst_type = utils.get_centroids(true_inst_mask, true_inst_type_img)  # 实例的质心
        if true_centroid.shape[0] == 0:  # 如果没有质心
            true_centroid = np.array([[0, 0]])
            true_inst_type = np.array([0])

        pred_inst_mask = masks_pred[i].astype("int32")
        pred_inst_type_img = classes_pred[i].astype("int32")
        pred_centroid, pred_inst_type = utils.get_centroids(pred_inst_mask, pred_inst_type_img) # 实例的质心
        if pred_centroid.shape[0] == 0:  # 这里只是用来判断有没有实例的
            pred_centroid = np.array([[0, 0]])
            pred_inst_type = np.array([0])

        # ! if take longer than 1min for 1000 vs 1000 pairing, sthg is wrong with coord
        paired, unpaired_true, unpaired_pred = pair_coordinates(true_centroid, pred_centroid, 12)  # ！！！根据质心进行配对

        # * Aggreate information
        # get the offset as each index represent 1 independent instance
        true_idx_offset = (
            true_idx_offset + true_inst_type_all[-1].shape[0] if i != 0 else 0
        )
        pred_idx_offset = (
            pred_idx_offset + pred_inst_type_all[-1].shape[0] if i != 0 else 0
        )  # 这里的偏移量是针对多张图像放在一个数组中操作
        true_inst_type_all.append(true_inst_type)
        pred_inst_type_all.append(pred_inst_type)

        # increment the pairing index statistic
        if paired.shape[0] != 0:  # ! sanity
            paired[:, 0] += true_idx_offset
            paired[:, 1] += pred_idx_offset
            paired_all.append(paired)

        unpaired_true += true_idx_offset
        unpaired_pred += pred_idx_offset
        unpaired_true_all.append(unpaired_true)
        unpaired_pred_all.append(unpaired_pred)  # 这里对于每个图像循环结束

    paired_all = np.concatenate(paired_all, axis=0)
    unpaired_true_all = np.concatenate(unpaired_true_all, axis=0)
    unpaired_pred_all = np.concatenate(unpaired_pred_all, axis=0)
    true_inst_type_all = np.concatenate(true_inst_type_all, axis=0)
    pred_inst_type_all = np.concatenate(pred_inst_type_all, axis=0)

    paired_true_type = true_inst_type_all[paired_all[:, 0]]  # 根据索引取出对应位置的类别
    paired_pred_type = pred_inst_type_all[paired_all[:, 1]]
    unpaired_true_type = true_inst_type_all[unpaired_true_all]
    unpaired_pred_type = pred_inst_type_all[unpaired_pred_all]

    ###
    def _f1_type(paired_true, paired_pred, unpaired_true, unpaired_pred, type_id, w):
        type_samples = (paired_true == type_id) | (paired_pred == type_id)

        paired_true = paired_true[type_samples]
        paired_pred = paired_pred[type_samples]

        tp_dt = ((paired_true == type_id) & (paired_pred == type_id)).sum()
        tn_dt = ((paired_true != type_id) & (paired_pred != type_id)).sum()
        fp_dt = ((paired_true != type_id) & (paired_pred == type_id)).sum()
        fn_dt = ((paired_true == type_id) & (paired_pred != type_id)).sum()

        if not exhaustive:
            ignore = (paired_true == -1).sum()
            fp_dt -= ignore

        fp_d = (unpaired_pred == type_id).sum()
        fn_d = (unpaired_true == type_id).sum()

        f1_type = (2 * (tp_dt + tn_dt)) / (
                2 * (tp_dt + tn_dt)
                + w[0] * fp_dt
                + w[1] * fn_dt
                + w[2] * fp_d
                + w[3] * fn_d
        )
        return f1_type

    # overall
    # * quite meaningless for not exhaustive annotated dataset
    w = [1, 1]
    tp_d = paired_pred_type.shape[0]
    fp_d = unpaired_pred_type.shape[0]
    fn_d = unpaired_true_type.shape[0]

    tp_tn_dt = (paired_pred_type == paired_true_type).sum()
    fp_fn_dt = (paired_pred_type != paired_true_type).sum()

    if not exhaustive:
        ignore = (paired_true_type == -1).sum()
        fp_fn_dt -= ignore

    f1_d = 2 * tp_d / (2 * tp_d + w[0] * fp_d + w[1] * fn_d)  # 这个是Fd
    acc_type = tp_tn_dt / (tp_tn_dt + fp_fn_dt)  # 这个是分类准确率

    w = [2, 2, 1, 1]

    if type_uid_list is None:
        type_uid_list = np.unique(true_inst_type_all).tolist()

    results_list = [f1_d * acc_type]
    for type_uid in type_uid_list:
        f1_type = _f1_type(
            paired_true_type,
            paired_pred_type,
            unpaired_true_type,
            unpaired_pred_type,
            type_uid,
            w,
        )
        results_list.append(f1_type)

    np.set_printoptions(formatter={"float": "{: 0.5f}".format})
    return results_list

def pair_coordinates(setA, setB, radius):
    """Use the Munkres or Kuhn-Munkres algorithm to find the most optimal
    unique pairing (largest possible match) when pairing points in set B
    against points in set A, using distance as cost function.

    Args:
        setA, setB: np.array (float32) of size Nx2 contains the of XY coordinate
                    of N different points
        radius: valid area around a point in setA to consider
                a given coordinate in setB a candidate for match
    Return:
        pairing: pairing is an array of indices
        where point at index pairing[0] in set A paired with point
        in set B at index pairing[1]
        unparedA, unpairedB: remaining poitn in set A and set B unpaired

    """
    # * Euclidean distance as the cost matrix  # 计算点之间的欧式距离
    setA_tile = np.expand_dims(setA, axis=1)
    setB_tile = np.expand_dims(setB, axis=0)
    setA_tile = np.repeat(setA_tile, setB.shape[0], axis=1)
    setB_tile = np.repeat(setB_tile, setA.shape[0], axis=0)
    pair_distance = (setA_tile - setB_tile) ** 2
    # set A is row, and set B is paired against set A
    pair_distance = np.sqrt(np.sum(pair_distance, axis=-1))

    # * Munkres pairing with scipy library
    # the algorithm return (row indices, matched column indices)
    # if there is multiple same cost in a row, index of first occurence
    # is return, thus the unique pairing is ensured
    indicesA, paired_indicesB = linear_sum_assignment(pair_distance)

    # extract the paired cost and remove instances
    # outside of designated radius
    pair_cost = pair_distance[indicesA, paired_indicesB]

    pairedA = indicesA[pair_cost <= radius]
    pairedB = paired_indicesB[pair_cost <= radius]

    unpairedA = [idx for idx in range(setA.shape[0]) if idx not in list(pairedA)]
    unpairedB = [idx for idx in range(setB.shape[0]) if idx not in list(pairedB)]

    pairing = np.array(list(zip(pairedA, pairedB)))
    unpairedA = np.array(unpairedA, dtype=np.int64)
    unpairedB = np.array(unpairedB, dtype=np.int64)

    return pairing, unpairedA, unpairedB


def refine_masks(M):
    masks=[]
    for mask in M:
        mask_refine = np.zeros_like(mask)
        mask_ids = np.unique(mask)
        for mask_id in range(1, len(mask_ids)):  # 默认0为背景
            proced_id_0 = mask_ids[mask_id]
            mask_refine[mask == proced_id_0] = mask_id
        masks.append(mask_refine)
    return masks
