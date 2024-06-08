import os
import sys
import uuid
import torch
import numpy as np
import random
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

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
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--model-type', '-t', type=str, default="custom", choices=["custom", "gs3d", "gs2d", "of"])
    parser.add_argument('--sugar', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Initialize system state (RNG)
    # safe_state(args.quiet)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    # # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if args.model_type == "custom":
        from trainers.custom import training
        training(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
        )
    if args.model_type == "gs3d":
        from trainers.gs3d import training_gs3d
        training_gs3d(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
        )
    if args.model_type == "of":
        from trainers.of import training_of
        training_of(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
            args.debug_from,
        )
    if args.model_type == "gs2d":
        from trainers.gs2d import training_gs2d
        training_gs2d(
            lp.extract(args),
            op.extract(args),
            pp.extract(args),
            args.test_iterations,
            args.save_iterations,
            args.checkpoint_iterations,
            args.start_checkpoint,
        )

    if args.sugar:

        from sugar.sugar_utils.general_utils import str2bool
        from sugar.sugar_trainers.coarse_density import coarse_training_with_density_regularization
        from sugar.sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
        from sugar.sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
        from sugar.sugar_trainers.refine import refined_training
        from sugar.sugar_extractors.refined_mesh import extract_mesh_and_texture_from_refined_sugar

        scene_path = args.source_path
        checkpoint_path = args.model_path if args.model_path.endswith('/') else args.model_path + '/'
        iteration_to_load = 7000
        regularization_type = "sdf" # sdf/density coarse SuGar
        surface_level = 0.3
        n_vertices_in_mesh = 1_000_000
        bboxmin = None
        bboxmax = None
        center_bbox = True
        gaussians_per_triangle = 1
        refinement_iterations = 15_000
        export_uv_textured_mesh = True
        square_size = 10
        postprocess_mesh = False
        postprocess_density_threshold = 0.1
        postprocess_iterations = 5
        export_ply = True
        low_poly = False
        high_poly = False
        refinement_time = None # short/medium/long
        eval_split = True
        gpu = 0
        white_background = False

        if low_poly:
            n_vertices_in_mesh = 200_000
            gaussians_per_triangle = 6
        if high_poly:
            n_vertices_in_mesh = 1_000_000
            gaussians_per_triangle = 1
            print('Using high poly config.')
        if refinement_time == 'short':
            refinement_iterations = 2_000
            print('Using short refinement time.')
        if refinement_time == 'medium':
            refinement_iterations = 7_000
            print('Using medium refinement time.')
        if refinement_time == 'long':
            refinement_iterations = 15_000
            print('Using long refinement time.')
        if export_uv_textured_mesh:
            print('Will export a UV-textured mesh as an .obj file.')
        if export_ply:
            print('Will export a ply file with the refined 3D Gaussians at the end of the training.')

        # ----- Optimize coarse SuGaR -----
        coarse_args = AttrDict({
            'checkpoint_path': checkpoint_path,
            'scene_path': scene_path,
            'iteration_to_load': iteration_to_load,
            'output_dir': None,
            'eval': eval_split,
            'estimation_factor': 0.2,
            'normal_factor': 0.2,
            'gpu': gpu,
            'white_background': white_background,
        })
        if regularization_type == 'sdf':
            coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args)
        elif regularization_type == 'density':
            coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
        else:
            raise ValueError(f'Unknown regularization type: {regularization_type}')


        # ----- Extract mesh from coarse SuGaR -----
        coarse_mesh_args = AttrDict({
            'scene_path': scene_path,
            'checkpoint_path': checkpoint_path,
            'iteration_to_load': iteration_to_load,
            'coarse_model_path': coarse_sugar_path,
            'surface_level': surface_level,
            'decimation_target': n_vertices_in_mesh,
            'mesh_output_dir': None,
            'bboxmin': bboxmin,
            'bboxmax': bboxmax,
            'center_bbox': center_bbox,
            'gpu': gpu,
            'eval': eval_split,
            'use_centers_to_extract_mesh': False,
            'use_marching_cubes': False,
            'use_vanilla_3dgs': False,
        })
        coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]

        # ----- Refine SuGaR -----
        refined_args = AttrDict({
            'scene_path': scene_path,
            'checkpoint_path': checkpoint_path,
            'mesh_path': coarse_mesh_path,      
            'output_dir': None,
            'iteration_to_load': iteration_to_load,
            'normal_consistency_factor': 0.1,    
            'gaussians_per_triangle': gaussians_per_triangle,        
            'n_vertices_in_fg': n_vertices_in_mesh,
            'refinement_iterations': refinement_iterations,
            'bboxmin': bboxmin,
            'bboxmax': bboxmax,
            'export_ply': export_ply,
            'eval': eval_split,
            'gpu': gpu,
            'white_background': white_background,
        })
        refined_sugar_path = refined_training(refined_args)


        # ----- Extract mesh and texture from refined SuGaR -----
        if export_uv_textured_mesh:
            refined_mesh_args = AttrDict({
                'scene_path': scene_path,
                'iteration_to_load': iteration_to_load,
                'checkpoint_path': checkpoint_path,
                'refined_model_path': refined_sugar_path,
                'mesh_output_dir': None,
                'n_gaussians_per_surface_triangle': gaussians_per_triangle,
                'square_size': square_size,
                'eval': eval_split,
                'gpu': gpu,
                'postprocess_mesh': postprocess_mesh,
                'postprocess_density_threshold': postprocess_density_threshold,
                'postprocess_iterations': postprocess_iterations,
            })
            refined_mesh_path = extract_mesh_and_texture_from_refined_sugar(refined_mesh_args)

    # All done
    print("\nTraining complete.")
