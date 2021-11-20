import numpy as np
import pickle
import torch
from torch.nn import Module

class SMPLModel(Module):
    def __init__(self, device=None, model_path='./model_lsp.pkl',
                 dtype=torch.float32, simplify=False, batch_size=1):
        super(SMPLModel, self).__init__()
        self.dtype = dtype
        self.simplify = simplify
        self.batch_size = batch_size
        with open(model_path, 'rb') as f:
            params = pickle.load(f, encoding='latin')
        self.J_regressor = torch.from_numpy(
            np.array(params['J_regressor'].todense())
        ).type(self.dtype)
        # 20190330: lsp 14 joint regressor
        self.joint_regressor = torch.from_numpy(
            np.load('J_regressor_lsp.npy')).type(self.dtype)

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

def cam_mat():
    exmats = []
    inmats = []
    #0
    exmats.append([
        [0.166605640169249, -0.293385695586161, 0.9413646446989706, 11.909454043496599],
        [-0.27117043600715557, -0.9315278019459141, -0.24232735882995768, -5.446204352747883 ],
        [0.9480027190359128, -0.21489715639635254, -0.23475531276901432, 26.467684858689605 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2398.95250917519, 0.0, 1022.838 ],
        [0.0, 2398.95250917519, 767.0612 ],
        [0.0, 0.0, 1.0 ]
    ])
    #1
    exmats.append([
        [0.11625639282071201, 0.19002593042350036, -0.9748715796939615, -13.699860315783948],
        [0.022525689371237143, -0.9817790714848064, -0.18868611001550173, -2.474198516323067 ],
        [-0.9929637679425504, -2.3687855259207238e-05, -0.1184185584788912, 25.431383471774243 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.46884683428, 0.0, 1036.4757 ],
        [0.0, 2418.46884683428, 762.90744 ],
        [0.0, 0.0, 1.0 ]
    ])
    #2
    exmats.append([
        [0.8345846593841691, 0.049668914036733444, -0.5486359861502169, -8.614951133534133 ],
        [-0.005165605164969915, -0.9951776379569582, -0.09795297562447698, -1.77651919492546 ],
        [-0.5508554827210936, 0.08458408768096592, -0.8303034200021636, 19.380850499269233 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2404.8628440717, 0.0, 1022.42111 ],
        [0.0, 2404.8628440717, 774.58679 ],
        [0.0, 0.0, 1.0 ]
    ])
    #3
    exmats.append([
        [0.8433363166637616, -0.11168498710159291, 0.5256522811251901, 6.498464239351408 ],
        [-0.0623025480207155, -0.9918890694603475, -0.110790190879934, -3.3095316017484615 ],
        [0.5337623530243822, 0.06068391501215786, -0.8434543336498258, 21.50535934865188 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.98884925097, 0.0, 1023.98034 ],
        [0.0, 2418.98884925097, 774.29853 ],
        [0.0, 0.0, 1.0 ]
    ])
    #4
    exmats.append([
        [-0.7102084639080114, 0.13673389372553238, -0.6905850998248353, -9.70675844846761 ],
        [0.0329757095646321, -0.9734186163315597, -0.2266468617869237, -3.9977853072052567 ],
        [-0.7032187002434819, -0.18373905324074874, 0.6868212430771649, 40.02720817109032 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.36776172459, 0.0, 1034.79508],
        [0.0, 2418.36776172459, 766.41563 ],
        [0.0, 0.0, 1.0 ]
    ])
    #5
    exmats.append([
        [-0.8720753788586445, -0.07631500400889554, 0.4833844781865401, 5.896196103261941 ],
        [-0.03625587837246755, -0.9749757270196461, -0.2193350017802762, -3.497412012631528 ],
        [0.48802668459008713, -0.20880218362276573, 0.8474854590152994, 42.31393611674388 ],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2390.21408950408, 0.0, 1024.318747 ],
        [0.0, 2390.21408950408, 772.82566 ],
        [0.0, 0.0, 1.0 ]
    ])
    return exmats, inmats

if __name__ == '__main__':

    def read_point(file_name):
        face = []
        f = open(file_name)
        line = f.readline()
        while line:
            line_data = line.split()
            if line_data[0] == 'f':
                face.append([float(line_data[1]),float(line_data[2]),float(line_data[3])])
            #color.append([float(line_data[4]),float(line_data[5]),float(line_data[6])])
            line = f.readline()
        f.close()
        return face#,color
    faces = read_point('template.obj')

    from scipy.spatial.transform import Rotation as R
    exmats, inmatx = cam_mat()

    RRR = np.array([
        [0.99994531,  0.01038064,  0.00127162, 0.00730746],
        [-0.01038064,  0.97037788,  0.24136904, 1.38704965],
        [0.00127162, -0.24136904,  0.97043257, 5.57668113],
        [0.0, 0.0, 0.0, 1.0]
        ])
    
    exmats_new = []
    for exmat in exmats:
        exmat = np.array(exmat)
        exmat[:3, 3] = exmat[:3, 3] / 7
        exmat[:3, :3] = np.dot(exmat[:3, :3], np.linalg.inv(RRR[:3,:3]))
        exmat[:3, 3] = exmat[:3, 3] - np.dot(exmat[:3, :3], RRR[:3, 3][:,None])[:,0]
        exmats_new.append(exmat)
    idx = 0
    for mat in exmats_new:
        np.savetxt(R'E:\Human-Training-v3.3\VCL Occlusion/3DOH50K_Parameters'+str(idx)+'.txt', mat)
        idx+=1        

    smpl_data = SMPLModel()
    import glob
    import os
    save_data = []
    save_data1 = []
    save_data2 = []
    cam_joint = []
    cam_joint1 = []
    path = R'E:\Human-Training-v3.2\原始素材 VCL Occlusion\params'
    dirs_path = glob.glob(os.path.join(path, '*'))
    i = 0

    save_path = R'E:\Human-Training-v3.3\VCL Occlusion\params'

    for dir_path in dirs_path[39:]:
        frames_dir = glob.glob(os.path.join(dir_path, '*'))
        
        dir_name_yy = os.path.basename(dir_path)
        if not os.path.exists(os.path.join(save_path,dir_name_yy)):
            os.makedirs(os.path.join(save_path,dir_name_yy))
        
        for frame_path in frames_dir:
            frames = glob.glob(os.path.join(frame_path, '*'))

            frame_name_yy = os.path.basename(frame_path)
            if not os.path.exists(os.path.join(save_path,dir_name_yy, frame_name_yy)):
                os.makedirs(os.path.join(save_path,dir_name_yy, frame_name_yy))

            for frame in frames:

                pkl_name_yy = os.path.basename(frame)

                with open(frame, 'rb') as f:
                    data = pickle.load(f)
                scale = data['scale'][0]
                
                vs, js = smpl_data(betas=torch.tensor(data['betas']), thetas=torch.tensor(data['pose']), trans=torch.tensor(data['transl']), scale=torch.tensor(data['scale']), gR=None, lsp=False)
                js = js.squeeze(0).numpy()
                vs = vs.squeeze(0).numpy()
                v1, j1 = smpl_data(betas=torch.tensor(data['betas']), thetas=torch.tensor(data['pose']), trans=torch.tensor([0,0,0]), scale=torch.tensor(1), gR=None, lsp=False)
                j1 = j1.squeeze(0).numpy()
                v1 = v1.squeeze(0).numpy()

                vs = vs / data['scale'][0]
                trans = (vs-v1).mean(axis = 0)
                v2, j2 = smpl_data(betas=torch.tensor(data['betas']), thetas=torch.tensor(data['pose']), trans=torch.tensor(trans), scale=torch.tensor(1), gR=None, lsp=False)
                j2 = j2.squeeze(0).numpy()
                v2 = v2.squeeze(0).numpy()

                v_root = j2[0] - trans

                data['pose'][0,:3] = (R.from_matrix(RRR[:3,:3])*R.from_rotvec(data['pose'][0,:3])).as_rotvec()
                trans = np.dot(RRR[:3,:3], trans[:,None])[:,0] + RRR[:3,3] + np.dot(RRR[:3,:3], v_root[:,None])[:,0] - v_root
                data['transl'] = trans
                data['scale'] = [1]
                with open(os.path.join(save_path,dir_name_yy, frame_name_yy, pkl_name_yy),'wb') as f:
                    pickle.dump(data, f)

                #v2, j2 = smpl_data(betas=torch.tensor(data['betas']), thetas=torch.tensor(data['pose']), trans=torch.tensor(data['transl']), scale=torch.tensor(1), gR=None, lsp=False)
                #v2 = v2.squeeze(0).numpy()
                # f = open('./data/smpl' + str(i) + '.obj', 'w')
                # for data in v2:
                #     f.write('v ' + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')
                # for face in faces:
                #     f.write('f ' + str(int(face[0])) + ' ' + str(int(face[1])) + ' ' + str(int(face[2])) + '\n')
                # f.close()
                # i += 1
                # j2 = j2.squeeze(0).numpy()
                # save_data.append(js[10])
                # save_data.append(js[11])
                # save_data1.append(j2[10])
                # save_data1.append(j2[11])
                # for exmat in exmats:
                #     exmat = np.array(exmat)
                #     exmat[:3, 3] = exmat[:3, 3] / 7
                #     cam_joint1.append(np.dot(exmat[:3, :3], js[10][:, None])[:,0] + exmat[:3,3])
                # for exmat in exmats_new:
                #     cam_joint.append(np.dot(exmat[:3, :3], j2[10][:, None])[:,0] + exmat[:3,3])
            
    # f = open('jointscam1.obj', 'w')
    # for data in cam_joint:
    #     f.write('v ' + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')
    # f.close()
    # f = open('jointscam2.obj', 'w')
    # for data in cam_joint1:
    #     f.write('v ' + str(data[0]) + ' ' + str(data[1]) + ' ' + str(data[2]) + '\n')
    # f.close()