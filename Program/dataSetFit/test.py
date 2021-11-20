import numpy as np
import pickle
import torch
from torch.nn import Module
import cv2
from utils.rotate_utils import *
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
        [1.647480490561696320e-01, -5.920818220871540416e-02, 9.845569897604451448e-01, -3.708289009933127822e+00],
        [-2.811336085316997746e-01, -9.596093715587274975e-01, -1.066521388759986144e-02, 6.145275193409486247e-01],
        [9.454215868018357449e-01, -2.750349869118076662e-01, -1.747391877108037395e-01, 5.130141119639111125e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2398.95250917519, 0.0, 1022.838 ],
        [0.0, 2398.95250917519, 767.0612 ],
        [0.0, 0.0, 1.0 ]
    ])
    #1
    exmats.append([
        [1.169829664011515363e-01, -5.211367336456138033e-02, -9.917656723566262711e-01, 3.645067412210982738e+00],
        [1.209302582118892445e-02, -9.984735091982360755e-01, 5.389256940501588322e-02, 7.308453564137551428e-01],
        [-9.930602935387384811e-01, -1.829795964544415898e-02, -1.161741739509442317e-01, 4.313558032242152507e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.46884683428, 0.0, 1036.4757 ],
        [0.0, 2418.46884683428, 762.90744 ],
        [0.0, 0.0, 1.0 ]
    ])
    #2
    exmats.append([
        [8.343569602736718993e-01, -9.288964915209084461e-02, -5.433414973966331774e-01, 1.922080496119266790e+00],
        [-1.562046259778864139e-02, -9.892875588318166269e-01, 1.451417443948225250e-01, 3.091074245728276382e-01],
        [-5.510031454500767811e-01, -1.126127784505883689e-01, -8.268699340501268757e-01, 7.540108835029387890e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2404.8628440717, 0.0, 1022.42111 ],
        [0.0, 2404.8628440717, 774.58679 ],
        [0.0, 0.0, 1.0 ]
    ])
    #3
    exmats.append([
        [8.427992611801159439e-01, 9.745173894998002129e-03, 5.381397879720993815e-01, -2.092357728622498136e+00],
        [-7.273646728962220032e-02, -9.886017929427162176e-01, 1.318176779366164730e-01, 1.638759037450721046e-01],
        [5.332905530620675183e-01, -1.502382289085486822e-01, -8.324840328678929646e-01, 7.919183073224576042e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.98884925097, 0.0, 1023.98034 ],
        [0.0, 2418.98884925097, 774.29853 ],
        [0.0, 0.0, 1.0 ]
    ])
    #4
    exmats.append([
        [-7.096283959244410466e-01, -2.662989747006347385e-02, -7.040727099424972657e-01, 2.581831788415241302e+00],
        [2.258099001661120070e-02, -9.996317365593113680e-01, 1.504955298584556567e-02, 7.313350957474699099e-01],
        [-7.042142010016161358e-01, -5.219068262047182882e-03, 7.099683983561549949e-01, 1.771290349772403516e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2418.36776172459, 0.0, 1034.79508],
        [0.0, 2418.36776172459, 766.41563 ],
        [0.0, 0.0, 1.0 ]
    ])
    #5
    exmats.append([
        [-8.722052076351175520e-01, 5.167235677460774168e-02, 4.864031764962798432e-01, -1.935500206613752550e+00],
        [-4.665367775368382819e-02, -9.986591974502415647e-01, 2.243302256968168762e-02, 7.607987090046788303e-01],
        [4.869101687488225916e-01, -3.126298573727731567e-03, 8.734464528452595689e-01, 1.174693919790863461e+00],
        [0.0, 0.0, 0.0, 1.0]
    ])
    inmats.append([
        [2390.21408950408, 0.0, 1024.318747 ],
        [0.0, 2390.21408950408, 772.82566 ],
        [0.0, 0.0, 1.0 ]
    ])
    return exmats, inmats
def cam_mat1():
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
def Camera_project(points, externalMat=None, internalMat=None):
    if externalMat is None:
        return points
    else:
        pointsInCamera = np.dot(externalMat, np.row_stack((points.T, np.ones(points.__len__()))))
        if internalMat is None:
            return pointsInCamera[:3,:].T
        else:
            pointsInImage = np.dot(internalMat, pointsInCamera[:3,:]) / pointsInCamera[2,:][None,:]
            return pointsInImage[:2,:].T
pkl = R'E:\Human-Training-v3.2\原始素材 VCL Occlusion\params\0000\00005/000.pkl'
frame = R'E:\Human-Training-v3.3\VCL Occlusion/00000.jpg'
matidx = 5
smpl_data = SMPLModel()
exmats, inmats = cam_mat1()
with open(pkl, 'rb') as f:
    data = pickle.load(f)
scale = data['scale'][0]
                
vs, js = smpl_data(betas=torch.tensor(data['betas']), thetas=torch.tensor(data['pose']), trans=torch.tensor(data['transl'].astype(np.float32)), scale=torch.tensor(data['scale']), gR=None, lsp=False)
js = js.squeeze(0).numpy()
vs = vs.squeeze(0).numpy()
                
exmat = np.array(exmats[matidx])
vs_2d = Camera_project(vs, exmat, np.array(inmats[matidx]))
img = cv2.imread(frame)
for v in vs_2d:
    cv2.circle(img, (int(v[0]), int(v[1])),1,(0,0,255),3)
cv2.imshow('1', img)
cv2.waitKey(0)
print(0)