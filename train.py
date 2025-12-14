#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
import wandb
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
import logging
from tqdm.auto import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def setup_file_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 기존 핸들러 제거
    while logger.handlers:
        logger.handlers.pop()

    # 파일 핸들러만 추가
    fh = logging.FileHandler("memlog.txt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    ))

    logger.addHandler(fh)
    return logger

logger = setup_file_logger()

def memlog(tag=""):
    alloc = torch.cuda.memory_allocated() / 1024**2
    resa  = torch.cuda.memory_reserved() / 1024**2

    logger.info(
        f"[{tag}] alloc={alloc:.1f}MB | reserved={resa:.1f}MB"
    )

def tensor_stats(t):
    return f"mean={t.mean():.3e}, std={t.std():.3e}, min={t.min():.3e}, max={t.max():.3e}"

def log_tensor_stats(model, prefix=""):
    for name, t in model.__dict__.items():
        if torch.is_tensor(t):
            logger.debug(f"{prefix}{name}: {tensor_stats(t)}")

    s = model.get_scaling
    logger.debug(f"{prefix}scaling: {tensor_stats(s)}")

def print_grad_tensors(model):
    for name, t in model.__dict__.items():
        if torch.is_tensor(t) and t.grad_fn is not None:
            logger.warning(f"grad_fn alive: {name} shape={t.shape} fn={t.grad_fn}")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    wandb.init(
        project="gaussian-splatting",
        name=f"{dataset.model_path.split('/')[-1]}-{opt.iterations}iters",
        config={
            "iterations": opt.iterations,
            "position_lr": opt.position_lr_init,
            "feature_lr": opt.feature_lr,
            "scaling_lr": opt.scaling_lr,
            "opacity_lr": opt.opacity_lr,
            "densify_until": opt.densify_until_iter,
            "model_path": dataset.model_path
        },
        settings=wandb.Settings(_disable_stats=False)  # ⬅️ 이것만 넣으면 자동 GPU 모니터링 켜짐
    )
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE, return_err=True)
        image, viewspace_point_tensor, visibility_filter, radii, err_img = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["err"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # -----------------------------------------------------------
        # Loss
        # -----------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value) 

        rendered_mask = (gaussians.denom > 0).squeeze(-1)

        #sparsity warmup
        target_lambda1 = 1e-4
        warmup_start_iter = 10000
        warmup_duration = 2000
        if iteration< warmup_start_iter:
            current_lambda1 = 0.0
        elif iteration < (warmup_start_iter + warmup_duration):
            prog = (iteration - warmup_start_iter) / warmup_duration
            current_lambda1 = target_lambda1*prog
        else:
            current_lambda1 = target_lambda1
        # L_aux 
        alpha = 1e-6
        per_pix_err = (image - gt_image).abs().mean(0, keepdim=True).detach()  # [1,H,W], 
        L_aux = (per_pix_err * err_img).sum()              # 스칼라

        # sparsity regularization
        s_k = gaussians.get_s_k  # [N,1]
        L_sp = torch.mean(s_k) 

        #variance regularization
        count = gaussians.E_k_count[rendered_mask]+1e-6
        avg_Ek = gaussians.E_k_sum[rendered_mask]/count  # [N,1]
        var_Ek = torch.clamp((gaussians.E_k_sq_sum[rendered_mask]/count) - (avg_Ek**2), min= 0).detach()
        L_var = torch.mean(s_k[rendered_mask]*var_Ek)

        full_var_Ek = torch.zeros((gaussians.get_xyz.shape[0],1), device="cuda")
        full_var_Ek[rendered_mask] = var_Ek
        

        #sparcity, variance를 sum으로 바꿔보기


        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        lambda2 = 1e-6
        loss_total = L_aux * alpha + loss+ L_sp * current_lambda1 + L_var * current_lambda1
        loss_total.backward()

        # -----------------------------------------------------------
        if iteration < opt.densify_until_iter:
            with torch.no_grad():
                E_k_view = (gaussians._e_k.grad / alpha).detach().clone()  # [N,1]
                gaussians.E_k_sum[rendered_mask] += E_k_view[rendered_mask]
                gaussians.E_k_sq_sum[rendered_mask] += E_k_view[rendered_mask]**2
                gaussians.E_k_count[rendered_mask] += 1

                #원래 로직
                mask = E_k_view > gaussians.E_k
                gaussians.E_k[mask] = E_k_view[mask]
        # -----------------------------------------------------------

        if iteration % 10 == 0: # ⬅️ 너무 자주 기록하면 오버헤드가 있으므로 조절
            log_data = {
                "iteration": iteration,
                "total_loss": loss.item(),
                "L1_loss": Ll1.item(),
                "total_points": gaussians.get_xyz.shape[0],
                "s_k/mean": s_k.mean().item(),
                "s_k/min": s_k.min().item(),
                "s_k/max": s_k.max().item(),
                "E_k/max": gaussians.E_k.max().item(),
                "E_k/mean": gaussians.E_k.mean().item(),
                "E_k_min": gaussians.E_k.min().item(),
                "Var_Ek/mean": var_Ek.mean().item() # 이건걍 넣은거
            }
            
            # (선택) PSNR도 함께 기록 (테스트 시)
            if iteration in testing_iterations:
                # ... (PSNR 계산 로직) ...
                # log_data["test_psnr"] = psnr_value 
                pass

            wandb.log(log_data)


        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                
                # cur_N = gaussians.get_xyz.shape[0]
                # allow_growth = int(cur_N*0.1)  # 현재 개수의 10% 만큼 허용

                # if hassattr(opt, "max_points"):
                #     allow_grow = (cur_N+allow_growth) < opt.max_points if hasattr(opt, "max_points")
                # else:
                #     allow_grow = (cur_N+allow_growth) < 


                # 일단 무조건 grow 허용... 사실  gt가 있는 거 아니면 max N 을 모르니까 결과가 안나와서
                # 발표때 어찌저찌 입털거아니면 max N 을 놓는게 의미가 없어보임
                allow_grow = True


                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    print(
                    "iter", iteration,
                    "N", gaussians.get_xyz.shape[0],
                    "E_k min", gaussians.E_k.min().item(), 
                    "E_k max", gaussians.E_k.max().item(),
                    "whohasmaxE_k", torch.argmax(gaussians.E_k).item()
                )

                    #❗ problem: the # of gaussians explodes...
                    # => we need to prune more aggressively
                    #optinon1: increase the opacity threshold for pruning
                    #option2: increase E_K threshold for densification

                    #original thresholds- opacity: 0.005, E_K: 0.1
                    if allow_grow:
                        E_k_thr = gaussians.nonlinear_error()
                        print(f"Densification E_k threshold: {E_k_thr}")
                        gaussians.densify_and_prune(E_k_thr,0.01,0.1, scene.cameras_extent, size_threshold, radii, full_var_Ek, 0.02, rule = 'both')

                    #allow_grow를 항상 true 로 놨으니까 걍 냅둠일단
                    # else:
                    #     gaussians.prune_points((gaussians.get_opacity < 0.01).squeeze())
                    gaussians.reset_Ek()

                            
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if gaussians._e_k.grad is not None:
                        gaussians._e_k.grad.zero_()
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    if gaussians._e_k.grad is not None:
                        gaussians._e_k.grad.zero_()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            if iteration % 50 == 0:
                logger.info(f"--- Iter {iteration} ---")
                memlog(f"iter {iteration}")
                log_tensor_stats(gaussians, prefix=f"[iter {iteration}] ")
                print_grad_tensors(gaussians)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
