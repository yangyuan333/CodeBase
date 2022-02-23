import smplx
import pickle as pkl
import torch
from torch.nn import Module
import numpy as np
import os
import sys
sys.path.append('./')
from utils.obj_utils import MeshData, write_obj
from scipy.spatial.transform import Rotation as R

class SMPLModel(Module):
    def __init__(self, device=None, model_path='./data/smplData/smpl/SMPL_MALE.pkl',
                 dtype=torch.float32, simplify=False, batch_size=1):
        super(SMPLModel, self).__init__()
        self.dtype = dtype
        self.simplify = simplify
        self.batch_size = batch_size
        with open(model_path, 'rb') as f:
            params = pkl.load(f, encoding='latin')
        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(self.dtype)
        # 20190330: lsp 14 joint regressor
        self.joint_regressor = torch.from_numpy(
            np.load('./data/smplData/smpl/J_regressor_lsp.npy')).type(self.dtype)

        self.weights = torch.from_numpy(params['weights']).type(self.dtype)
        self.posedirs = torch.from_numpy(params['posedirs']).type(self.dtype)
        self.v_template = torch.from_numpy(params['v_template']).type(self.dtype)
        self.shapedirs = torch.from_numpy(np.array(params['shapedirs'])).type(self.dtype)
        self.kintree_table = params['kintree_table']
        id_to_col = {self.kintree_table[1, i]: i
                     for i in range(self.kintree_table.shape[1])}
        self.parent = {
            i: id_to_col[self.kintree_table[0, i]]
            for i in range(1, self.kintree_table.shape[1])
        }
        self.faces = params['f']
        self.device = device if device is not None else torch.device('cpu')
        # add torch param
        mean_pose = np.zeros((24, 1))
        mean_shape = np.zeros((1, 10))

        pose = torch.from_numpy(mean_pose) \
            .type(dtype).to(device)
        pose = torch.nn.Parameter(pose, requires_grad=True)
        self.register_parameter('pose', pose)

        shape = torch.from_numpy(mean_shape) \
            .type(dtype).to(device)
        shape = torch.nn.Parameter(shape, requires_grad=True)
        self.register_parameter('shape', shape)

        transl = torch.zeros([3],dtype=dtype).to(device)
        transl = torch.nn.Parameter(transl, requires_grad=True)
        self.register_parameter('transl', transl)

        scale = torch.ones([1],dtype=dtype).to(device)
        scale = torch.nn.Parameter(scale, requires_grad=True)
        self.register_parameter('scale', scale)

        # vertex_ids = SMPL_VIDs['smpl']
        # self.vertex_joint_selector = VertexJointSelector(
        #             vertex_ids=vertex_ids, dtype=dtype)

        for name in ['J_regressor', 'joint_regressor', 'weights', 'posedirs', 'v_template', 'shapedirs']:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

    @torch.no_grad()
    def reset_params(self, **params_dict):
        for param_name, param in self.named_parameters():
            if param_name in params_dict:
                param[:] = torch.tensor(params_dict[param_name])
            else:
                param.fill_(0)

    @staticmethod
    def rodrigues(r):
        """
        Rodrigues' rotation formula that turns axis-angle tensor into rotation
        matrix in a batch-ed manner.
        Parameter:
        ----------
        r: Axis-angle rotation tensor of shape [batch_size * angle_num, 1, 3].
        Return:
        -------
        Rotation matrix of shape [batch_size * angle_num, 3, 3].
        """
        eps = r.clone().normal_(std=1e-8)
        theta = torch.norm(r + eps, dim=(1, 2), keepdim=True)  # dim cannot be tuple
        theta_dim = theta.shape[0]
        r_hat = r / theta
        cos = torch.cos(theta)
        z_stick = torch.zeros(theta_dim, dtype=r.dtype).to(r.device)
        m = torch.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
             -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), dim=1)
        m = torch.reshape(m, (-1, 3, 3))
        i_cube = (torch.eye(3, dtype=r.dtype).unsqueeze(dim=0) \
                  + torch.zeros((theta_dim, 3, 3), dtype=r.dtype)).to(r.device)
        A = r_hat.permute(0, 2, 1)
        dot = torch.matmul(A, r_hat)
        R = cos * i_cube + (1 - cos) * dot + torch.sin(theta) * m
        return R

    @staticmethod
    def with_zeros(x):
        """
        Append a [0, 0, 0, 1] tensor to a [3, 4] tensor.
        Parameter:
        ---------
        x: Tensor to be appended.
        Return:
        ------
        Tensor after appending of shape [4,4]
        """
        ones = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]], dtype=x.dtype
        ).expand(x.shape[0], -1, -1).to(x.device)
        ret = torch.cat((x, ones), dim=1)
        return ret

    @staticmethod
    def pack(x):
        """
        Append zero tensors of shape [4, 3] to a batch of [4, 1] shape tensor.
        Parameter:
        ----------
        x: A tensor of shape [batch_size, 4, 1]
        Return:
        ------
        A tensor of shape [batch_size, 4, 4] after appending.
        """
        zeros43 = torch.zeros(
            (x.shape[0], x.shape[1], 4, 3), dtype=x.dtype).to(x.device)
        ret = torch.cat((zeros43, x), dim=3)
        return ret

    def write_obj(self, verts, file_name):
        with open(file_name, 'w') as fp:
            for v in verts:
                fp.write('v %f %f %f\n' % (v[0], v[1], v[2]))

            for f in self.faces + 1:
                fp.write('f %d %d %d\n' % (f[0], f[1], f[2]))

    def visualize_model_parameters(self):
        self.write_obj(self.v_template, 'v_template.obj')

    '''
      _lR2G: Buildin function, calculating G terms for each vertex.
    '''

    def _lR2G(self, lRs, J, scale):
        batch_num = lRs.shape[0]
        lRs[:,0] *= scale
        results = []  # results correspond to G' terms in original paper.
        results.append(
            self.with_zeros(torch.cat((lRs[:, 0], torch.reshape(J[:, 0, :], (-1, 3, 1))), dim=2))
        )
        for i in range(1, self.kintree_table.shape[1]):
            results.append(
                torch.matmul(
                    results[self.parent[i]],
                    self.with_zeros(
                        torch.cat(
                            (lRs[:, i], torch.reshape(J[:, i, :] - J[:, self.parent[i], :], (-1, 3, 1))),
                            dim=2
                        )
                    )
                )
            )

        stacked = torch.stack(results, dim=1)
        deformed_joint = \
            torch.matmul(
                stacked,
                torch.reshape(
                    torch.cat((J, torch.zeros((batch_num, 24, 1), dtype=self.dtype).to(self.device)), dim=2),
                    (batch_num, 24, 4, 1)
                )
            )
        results = stacked - self.pack(deformed_joint)
        return results, lRs

    def theta2G(self, thetas, J, scale):
        batch_num = thetas.shape[0]
        lRs = self.rodrigues(thetas.view(-1, 1, 3)).reshape(batch_num, -1, 3, 3)
        return self._lR2G(lRs, J, scale)

    '''
      gR2G: Calculate G terms from global rotation matrices.
      --------------------------------------------------
      Input: gR: global rotation matrices [N * 24 * 3 * 3]
             J: shape blended template pose J(b)
    '''

    def gR2G(self, gR, J):
        # convert global R to local R
        lRs = [gR[:, 0]]
        for i in range(1, self.kintree_table.shape[1]):
            # Solve the relative rotation matrix at current joint
            # Apply inverse rotation for all subnodes of the tree rooted at current joint
            # Update: Compute quick inverse for rotation matrices (actually the transpose)
            lRs.append(torch.bmm(gR[:, self.parent[i]].transpose(1, 2), gR[:, i]))

        lRs = torch.stack(lRs, dim=1)
        return self._lR2G(lRs, J)

    def forward(self, betas=None, thetas=None, trans=None, scale=None, gR=None, lsp=False):

        """
              Construct a compute graph that takes in parameters and outputs a tensor as
              model vertices. Face indices are also returned as a numpy ndarray.

              20190128: Add batch support.
              20190322: Extending forward compatiability with SMPLModelv3

              Usage:
              ---------
              meshes, joints = forward(betas, thetas, trans): normal SMPL
              meshes, joints = forward(betas, thetas, trans, gR=gR):
                    calling from SMPLModelv3, using gR to cache G terms, ignoring thetas
              Parameters:
              ---------
              thetas: an [N, 24 * 3] tensor indicating child joint rotation
              relative to parent joint. For root joint it's global orientation.
              Represented in a axis-angle format.
              betas: Parameter for model shape. A tensor of shape [N, 10] as coefficients of
              PCA components. Only 10 components were released by SMPL author.
              trans: Global translation tensor of shape [N, 3].

              G, R_cube_big: (Added on 0322) Fix compatible issue when calling from v3 objects
                when calling this mode, theta must be set as None

              Return:
              ------
              A 3-D tensor of [N * 6890 * 3] for vertices,
              and the corresponding [N * 24 * 3] joint positions.
        """
        batch_num = betas.shape[0]
        if scale is None:
            scale = self.scale
        v_shaped = (torch.tensordot(betas, self.shapedirs, dims=([1], [2])) + self.v_template)
        J = torch.matmul(self.J_regressor, v_shaped)
        if gR is not None:
            G, R_cube_big = self.gR2G(gR, J)
        elif thetas is not None:
            G, R_cube_big = self.theta2G(thetas, J, scale)  # pre-calculate G terms for skinning
        else:
            raise (RuntimeError('Either thetas or gR should be specified, but detected two Nonetypes'))

        # (1) Pose shape blending (SMPL formula(9))
        if self.simplify:
            v_posed = v_shaped
        else:
            R_cube = R_cube_big[:, 1:, :, :]
            I_cube = (torch.eye(3, dtype=self.dtype).unsqueeze(dim=0) + \
                      torch.zeros((batch_num, R_cube.shape[1], 3, 3), dtype=self.dtype)).to(self.device)
            lrotmin = (R_cube - I_cube).reshape(batch_num, -1)
            v_posed = v_shaped + torch.tensordot(lrotmin, self.posedirs, dims=([1], [2]))

        # (2) Skinning (W)
        T = torch.tensordot(G, self.weights, dims=([1], [1])).permute(0, 3, 1, 2)
        rest_shape_h = torch.cat(
            (v_posed, torch.ones((batch_num, v_posed.shape[1], 1), dtype=self.dtype).to(self.device)), dim=2
        )
        v = torch.matmul(T, torch.reshape(rest_shape_h, (batch_num, -1, 4, 1)))
        v = torch.reshape(v, (batch_num, -1, 4))[:, :, :3]
        result = v + torch.reshape(trans, (batch_num, 1, 3))

        # estimate 3D joint locations
        # joints = torch.tensordot(result, self.joint_regressor, dims=([1], [0])).transpose(1, 2)
        if lsp:
            joints = torch.tensordot(result, self.joint_regressor.transpose(0, 1), dims=([1], [0])).transpose(1, 2)
        else:
            joints = torch.tensordot(result, self.J_regressor.transpose(0, 1), dims=([1], [0])).transpose(1, 2)
        return result, joints

def pkl2smpl(pklPath, mode='smpl', modelPath='./data/smplData/body_models', gender='male', ext='pkl', **config):
    with open(pklPath, 'rb') as file:
        data = pkl.load(file, encoding='iso-8859-1')

        if mode.lower() == 'smpl':
            model = smplx.create(modelPath,mode,gender=gender)
            output = model(
                betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                global_orient = torch.tensor(data['person00']['global_orient'][None,:].astype(np.float32)),
                transl = torch.tensor(data['person00']['transl'][None,:].astype(np.float32)))
        elif mode.lower() == 'smplx':
            model = smplx.create(modelPath, mode,
                                gender=gender, use_face_contour=False,
                                num_betas=data['person00']['betas'].shape[1],
                                num_pca_comps=data['person00']['left_hand_pose'].shape[0],
                                ext=ext)
            output = model(
                    betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                    global_orient = torch.tensor(data['person00']['global_orient'][None,:].astype(np.float32)),
                    body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                    left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
                    right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
                    transl = torch.tensor(data['person00']['transl'][None,:].astype(np.float32)),
                    jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))
    
    if 'savePath' in config:
        meshData = MeshData()
        meshData.vert = output.vertices.detach().cpu().numpy().squeeze()
        meshData.face = model.faces + 1
        write_obj(config['savePath'], meshData)
    return output.vertices.detach().cpu().numpy().squeeze(),output.joints.detach().cpu().numpy().squeeze(),model.faces + 1

def creatModel(mode='smpl', modelPath='./data/smplData/body_models', gender='male', ext='pkl', **config):
    if mode.lower() == 'smpl':
        return smplx.create(smplx.create(modelPath,mode,gender=gender))
    elif mode.lower() == 'smplx':
        return smplx.create(modelPath, mode,
                            gender=gender, use_face_contour=False,
                            num_betas=config['num_betas'] if 'num_betas' in config else 10,
                            num_pca_comps=config['num_pca_comps'] if 'num_pca_comps' in config else 12,
                            ext=ext)

def smplx2smpl(smplxMeshPath,smplPklSavePath,cfg_path=R'./data/smplData/smplx2smpl.yaml'):
    from transfer_model.data import build_dataloader
    from transfer_model.transfer_model import run_fitting
    from transfer_model.utils import read_deformation_transfer
    from transfer_model.config import read_yaml
    from smplx import build_layer
    from loguru import logger
    import tqdm
    exp_cfg = read_yaml(cfg_path)
    exp_cfg.datasets.mesh_folder.data_folder = smplxMeshPath
    exp_cfg.output_folder = smplPklSavePath
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)

    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    ## 自定义
    output_folder = os.path.expanduser(os.path.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder
    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(
        deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = os.path.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if os.path.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']

    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        var_dict = run_fitting(
            exp_cfg, batch, body_model, def_matrix, mask_ids)
        paths = batch['paths']

        for ii, path in enumerate(paths):
            _, fname = os.path.split(path)

            output_path = os.path.join(
                output_folder, fname.split('_')[0]+'_smpl.pkl')
            with open(output_path, 'wb') as f:
                pkl.dump(var_dict, f)

            output_path = os.path.join(
                output_folder, fname.split('_')[0]+'_smpl.obj')
            meshData = MeshData()
            meshData.vert = var_dict['vertices'][ii].detach().cpu().numpy()
            meshData.face = var_dict['faces']+1
            write_obj(output_path, meshData)

def applyRot2Smpl(pklPath,savePath,Rotm=np.eye(3,3),Tm=np.zeros(3),mode='smpl',gender='male'):
    if mode.lower() == 'smpl':
        model = creatModel(gender=gender)
        with open(pklPath,'rb') as file:
            data = pkl.load(file)
        output = model(
            betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
            body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
            global_orient = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
            transl = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)))
        js = output.joints.detach().cpu().numpy().squeeze()
        j0 = js[0]
        data['person00']['pose'][:3] = (R.from_matrix(Rotm)*R.from_rotvec(data['person00']['pose'][:3])).as_rotvec()
        data['person00']['global_orient'] = data['person00']['pose'][:3]
        data['person00']['transl'] = R.from_matrix(Rotm).apply(j0 + data['person00']['transl']) - j0 + Tm
        # data['person00']['transl'] = R.from_matrix(Rotm).apply(data['person00']['transl']) + Tm
        with open(savePath,'wb') as file:
            pkl.dump(data, file)
    elif mode.lower() == 'smplx':
        with open(pklPath,'rb') as file:
            data = pkl.load(file)
        model = creatModel(
            mode,
            gender=gender,
            num_betas=data['person00']['betas'].shape[1],
            num_pca_comps=data['person00']['left_hand_pose'].shape[0])

        output = model(
                betas = torch.tensor(data['person00']['betas'].astype(np.float32)),
                global_orient = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
                body_pose = torch.tensor(data['person00']['body_pose'][None,:].astype(np.float32)),
                left_hand_pose = torch.tensor(data['person00']['left_hand_pose'][None,:].astype(np.float32)),
                right_hand_pose = torch.tensor(data['person00']['right_hand_pose'][None,:].astype(np.float32)),
                transl = torch.tensor(np.array([[0.0,0.0,0.0]]).astype(np.float32)),
                jaw_pose = torch.tensor(data['person00']['jaw_pose'][None,:].astype(np.float32)))
        js = output.joints.detach().cpu().numpy().squeeze()
        j0 = js[0]
        data['person00']['global_orient'] = (R.from_matrix(Rotm)*R.from_rotvec(data['person00']['global_orient'])).as_rotvec()
        data['person00']['transl'] = R.from_matrix(Rotm).apply(j0 + data['person00']['transl']) - j0 + Tm
        # data['person00']['transl'] = R.from_matrix(Rotm).apply(data['person00']['transl']) + Tm
        with open(savePath,'wb') as file:
            pkl.dump(data, file)

def applyRot2Exmat(rotEx,TEx,Rotm=np.eye(3,3),Tm=np.zeros(3)):
    '''
    rot   : 3*3
    T     : 3
    rotEx : 3*3
    TEx   : 3
    '''
    rotExTem = np.dot(rotEx,np.linalg.inv(Rotm))
    TExTem = TEx - np.dot(rotExTem,Tm[:,None])[:,0]
    return rotExTem, TExTem