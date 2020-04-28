import matplotlib.pyplot as plt
import torch

import deepsmlm.generic.emitter
from deepsmlm.evaluation.evaluation import WeightedErrors
from deepsmlm.generic.plotting import frame_coord

from deepsmlm.evaluation import evaluation, match_emittersets


def log_frames(x, y_out, y_tar, weight, em_out, em_tar, tp, tp_match, logger, step, colorbar=True):

    r_ix = torch.randint(0, len(x), (1, )).long().item()
    assert x.dim() == 4

    # rm batch dimension, i.e. select one sample
    x = x[r_ix]
    y_out = y_out[r_ix]
    y_tar = y_tar[r_ix]
    weight = weight[r_ix]

    assert isinstance(em_tar, deepsmlm.generic.emitter.EmitterSet)
    em_tar = em_tar.get_subset_frame(r_ix, r_ix)
    em_out = em_out.get_subset_frame(r_ix, r_ix)
    em_tp = tp.get_subset_frame(r_ix, r_ix)
    em_tp_match = tp_match.get_subset_frame(r_ix, r_ix)

    # loop over all input channels
    for i, xc in enumerate(x):
        f_input = plt.figure()
        frame_coord.PlotFrameCoord(xc, pos_tar=em_tar.xyz_px, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('input/raw_input_ch_' + str(i), f_input, step)

    # loop over all output channels
    for i, yc in enumerate(y_out):
        f_out = plt.figure()
        frame_coord.PlotFrameCoord(yc, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('output/raw_output_ch_' + str(i), f_out, step)

    # record tar / output emitters
    tar_ch = (x.size(0) - 1) // 2

    f_em_out = plt.figure(figsize=(10, 8))
    frame_coord.PlotFrameCoord(x[tar_ch], pos_tar=em_tar.xyz_px, pos_out=em_out.xyz_px).plot()
    logger.add_figure('em_out/em_out_tar', f_em_out, step)

    f_em_out3d = plt.figure(figsize=(10, 8))
    frame_coord.PlotCoordinates3D(pos_tar=em_tar.xyz_px, pos_out=em_out.xyz_px).plot()
    logger.add_figure('em_out/em_out_tar_3d', f_em_out3d, step)

    f_match = plt.figure(figsize=(10, 8))
    frame_coord.PlotFrameCoord(x[tar_ch], pos_tar=em_tp_match.xyz_px, pos_out=em_tp.xyz_px, match_lines=True,
                               labels=('TP match', 'TP')).plot()
    logger.add_figure('em_out/em_match', f_match, step)

    f_match_3d = plt.figure(figsize=(10, 8))
    frame_coord.PlotCoordinates3D(pos_tar=em_tp_match.xyz_px, pos_out=em_tp.xyz_px, match_lines=True,
                                  labels=('TP match', 'TP')).plot()
    logger.add_figure('em_out/em_match_3d', f_match_3d, step)

    # loop over all target channels
    for i, yct in enumerate(y_tar):
        f_tar = plt.figure()
        frame_coord.PlotFrameCoord(yct, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('target/target_ch_' + str(i), f_tar, step)

    # loop over all weight channels
    for i, w in enumerate(weight):
        f_w = plt.figure()
        frame_coord.PlotFrameCoord(w, plot_colorbar_frame=colorbar).plot()
        logger.add_figure('weight/weight_ch_' + str(i), f_w, step)


def log_kpi(loss_scalar: float, loss_cmp: dict, eval_set: dict, logger, step):

    logger.add_scalar('loss/test_ep', loss_scalar, step)

    assert loss_cmp.dim() == 4
    for i in range(loss_cmp.size(1)):  # loop over all channels
        logger.add_scalar('loss/test_ep_oss_ch_' + str(i), loss_cmp[:, i].mean(), step)

    logger.add_scalar_dict('eval/', eval_set, step)


def log_dists(x, y_out, y_tar, weight, em_out, em_tar, tp, tp_match, logger, step):

    """Log z vs z_gt"""
    f_tar = plt.figure()
    plt.plot(tp_match.xyz_nm[:, 2], tp.xyz_nm[:, 2], 'x')
    plt.plot(tp_match.xyz_nm[:, 2], tp_match.xyz_nm[:, 2], 'r')
    plt.xlabel('z gt')
    plt.ylabel('z pred.')
    logger.add_figure('residuals/z_gt_pred', f_tar, step)


def log_train(*args):
    return


def post_process_log_test(loss_cmp, loss_scalar, x, y_out, y_tar, weight, em_tar, post_processor, matcher, logger, step):

    """Post-Process"""
    em_out = post_processor.forward(y_out)

    """Match and Evaluate"""
    tp, fp, fn, tp_match = matcher.forward(em_out, em_tar)
    result = evaluation.EvalSet(weighted_eval=WeightedErrors(mode='phot', reduction='gaussian')).forward(tp, fp, fn, tp_match)

    """Log"""
    # raw frames
    log_frames(x=x, y_out=y_out, y_tar=y_tar, weight=weight, em_out=em_out, em_tar=em_tar, tp=tp, tp_match=tp_match,
               logger=logger, step=step)

    # KPIs
    log_kpi(loss_scalar=loss_scalar, loss_cmp=loss_cmp, eval_set=result._asdict(), logger=logger, step=step)

    # distributions
    log_dists(x=x, y_out=y_out, y_tar=y_tar, weight=weight, em_out=em_out, em_tar=em_tar, tp=tp, tp_match=tp_match,
              logger=logger, step=step)

    return