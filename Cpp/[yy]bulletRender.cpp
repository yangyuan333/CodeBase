#define _USE_MATH_DEFINES
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS

#include <fstream>
#include<io.h>
#include <windows.h>
#include <filesystem>
#include <string>
#include <vector>
#include "../../CommonUtils/Viewer/Viewer_offscreen.h"
#include "../../CommonUtils/Viewer/Viewer_gui.h"
#include "../../CommonUtils/Models/TetMesh.h"
#include "../../CommonUtils/Loaders/mat_loader.hpp"
#include "../../CommonUtils/utils_func.hpp"
#include "../../CommonUtils/Loaders/mesh_loader.hpp"
#include "PhysicsWrapper.hpp"
#include "../../CommonUtils/Loaders/mesh_loader.hpp"
// for high level API
#include<bullet/SharedMemory/SharedMemoryPublic.h>
#include<bullet/SharedMemory/b3RobotSimulatorClientAPI_NoGUI.h>
#pragma comment(lib, "opencv_world451.lib")
#ifdef _DEBUG
#pragma comment(lib, "glfw3dll.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "opencv_world451d.lib")
#pragma comment(lib, "OpenMeshCored.lib")
#pragma comment(lib, "OpenMeshToolsd.lib")
#pragma comment(lib, "../../libs/CommonUtilsd.lib")

#pragma comment(lib, "BulletDynamics_Debug.lib")
#pragma comment(lib, "BulletSoftBody_Debug.lib")
#pragma comment(lib, "BulletCollision_Debug.lib")
#pragma comment(lib, "LinearMath_Debug.lib")

// for high level API
#pragma comment(lib, "ws2_32.lib") // TCP socket in BulletRobotics.lib
#pragma comment(lib, "BulletRobotics_Debug.lib")
#pragma comment(lib, "BulletFileLoader_Debug.lib")
#pragma comment(lib, "BulletWorldImporter_Debug.lib")
#pragma comment(lib, "Bullet3Common_Debug.lib")
#pragma comment(lib, "BulletInverseDynamics_Debug.lib")
#pragma comment(lib, "BulletInverseDynamicsUtils_Debug.lib")

#else
#pragma comment(lib, "glfw3dll.lib")
#pragma comment(lib, "glew32.lib")
#pragma comment(lib, "opengl32.lib")
#pragma comment(lib, "opencv_world451.lib")
#pragma comment(lib, "OpenMeshCore.lib")
#pragma comment(lib, "OpenMeshTools.lib")
#pragma comment(lib, "../../libs/CommonUtils.lib")

#pragma comment(lib, "BulletDynamics.lib")
#pragma comment(lib, "BulletSoftBody.lib")
#pragma comment(lib, "BulletCollision.lib")
#pragma comment(lib, "LinearMath.lib")

// for high level API
#pragma comment(lib, "ws2_32.lib") // TCP socket in BulletRobotics.lib
#pragma comment(lib, "BulletRobotics.lib")
#pragma comment(lib, "BulletFileLoader.lib")
#pragma comment(lib, "BulletWorldImporter.lib")
#pragma comment(lib, "Bullet3Common.lib")
#pragma comment(lib, "BulletInverseDynamics.lib")
#pragma comment(lib, "BulletInverseDynamicsUtils.lib")
#endif // _DEBUG

#define Physcap 1
#define Amass 2
#define AmassDof 3
using namespace cv;
using namespace Eigen;
using namespace std;
using namespace std::filesystem;

struct Config {
	int AmassInSmpl[19] = {
		1,4,7,2,5,8,
		3,6,9,12,15,
		13,16,18,20,
		14,17,19,21
	};
	int smplInAmassDof[24] = {
		0,1,16,31,5,20,35,9,24,39,13,28,43,51,64,47,55,68,59,72,63,76,63,76
	};
	int smplInAmassDofDim[72] = {
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			0,0,0,
			0,0,0,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			0,0,0,
			0,0,0,
			0,0,0,
			0,0,0,
	};
	int smplInPhyscap[24] = {
		0,2,5,9,13,17,21,25,29,33,37,43,49,52,55,59,63,67,71,75,79,83,87,91
	};
	int smplInPhyscapDofDim[72] = {
			1,1,1,
			1,1,1,
			1,1,1,
			0,0,0,
			1,0,0,
			1,0,0,
			0,0,0,
			1,1,1,
			1,1,1,
			0,0,0,
			0,0,1,
			0,0,1,
			1,1,1,
			0,1,1,
			0,1,1,
			0,0,0,
			0,1,1,
			0,1,1,
			1,1,0,
			1,1,0,
			0,0,0,
			0,0,0,
			0,0,0,
			0,0,0,
	};
	int smplInAmassOneDof[24] = {
		0,1,13,25,5,17,29,9,21,33,-1,-1,37,45,61,41,49,65,53,69,57,73,-1,-1
	};
	int smplInAmssOneDofDim[72] = {
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			0,0,0,
			0,0,0,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			1,1,1,
			0,0,0,
			0,0,0,
			0,0,0,
			0,0,0,
	};
	int skeletons[23][2] = {
		{0,1},
		{1,4},
		{4,7},
		{7,10},
		{0,2},
		{2,5},
		{5,8},
		{8,11},
		{0,3},
		{3,6},
		{6,9},
		{9,13},
		{13,16},
		{16,18},
		{18,20},
		{20,22},
		{9,14},
		{14,17},
		{17,19},
		{19,21},
		{21,23},
		{9,12},
		{12,15},
	};
	int skeletonNum = 23;
};

struct CamMat {
	Matrix3f inmat = Matrix3f::Zero(); Matrix4f exmat = Matrix4f::Zero();
	Vector3f trans = Vector3f::Zero(); Vector3f v = Vector3f::Zero();
	Vector2i windowSize = Vector2i(960, 960);
};

Eigen::Quaternionf AxisAngle2Quaternion(const Eigen::Vector3f& v)
{
	Eigen::AngleAxisf aa_in(v.stableNorm(), v.stableNormalized());
	Eigen::Quaternionf q_out(aa_in);
	return q_out;
}
Eigen::Vector3f Quaternion2AxisAngle(const Eigen::Quaternionf& q)
{
	Eigen::AngleAxisf aa_out(q);
	return(aa_out.axis() * aa_out.angle());
}
void Rmat2Euler(std::vector<float>& ordered_angles, const Eigen::Matrix3f& Rmat, const std::vector<Eigen::Vector3f> xyz_order) {
	std::vector<int> paras;
	std::vector<std::string> a_s{ "x", "y", "z" };
	for (auto& axis : xyz_order)
	{
		if (axis == Eigen::Vector3f::UnitX()) paras.push_back(0);
		else if (axis == Eigen::Vector3f::UnitY()) paras.push_back(1);
		else if (axis == Eigen::Vector3f::UnitZ()) paras.push_back(2);
	}

	if (paras.size() < 3) {
		if (std::find(paras.begin(), paras.end(), 0) == paras.end()) paras.push_back(0);
		if (std::find(paras.begin(), paras.end(), 1) == paras.end()) paras.push_back(1);
		if (std::find(paras.begin(), paras.end(), 2) == paras.end()) paras.push_back(2);
	}

	int i = paras[2], j = paras[1], k = paras[0];
	/*printf("ijk = %d - %d - %d \n", i, j, k);
	std::cout << Rmat << std::endl;*/
	Eigen::Vector3f angles;
	float cy = std::sqrt(Rmat(i, i) * Rmat(i, i) + Rmat(j, i) * Rmat(j, i));
	if (cy > 4e-5) {
		angles[2] = std::atan2(Rmat(k, j), Rmat(k, k));
		angles[1] = std::atan2(-Rmat(k, i), cy);
		angles[0] = std::atan2(Rmat(j, i), Rmat(i, i));
	}
	else {
		angles[2] = std::atan2(-Rmat(j, k), Rmat(j, j));
		angles[1] = std::atan2(-Rmat(k, i), cy);
		angles[0] = 0;
	}

	if ((i - j) * (j - k) * (k - i) == -2) { angles *= -1; /*parity = True*/ }

	//std::cout << "anlge res: " << angles.transpose() << std::endl;
	ordered_angles.clear();
	ordered_angles.resize(xyz_order.size());
	for (int ang_index = 0; ang_index < xyz_order.size(); ang_index++) { ordered_angles[ang_index] = angles[ang_index]; }
}
string zfill(string a, int num)
{
	while (a.size() < num) a = "0" + a;
	return a;
}
void load_cam_params(const string& filename, vector<Matrix3f>& intris, vector<Matrix4f>& extris)
{
	ifstream infile(filename);
	if (infile.is_open()) {
		while (!infile.eof()) {
			int camidx;
			float tmp;
			infile >> camidx;
			if (extris.size() <= camidx) {
				extris.resize(camidx + 1);
				extris[camidx] = Matrix4f::Identity();
				intris.resize(camidx + 1);
			}

			for (int i = 0; i < 3; i++)
				infile >> intris[camidx](i, 0) >> intris[camidx](i, 1) >> intris[camidx](i, 2);
			infile >> tmp >> tmp;
			for (int i = 0; i < 3; i++)
				infile >> extris[camidx](i, 0) >> extris[camidx](i, 1)
				>> extris[camidx](i, 2) >> extris[camidx](i, 3);
		}
		infile.close();
	}
	else {
		printf("can not read %s\n", filename.c_str());
		return;
	}
}
void getFolder(string path, vector<string>& folders)
{
	//文件句柄  
	intptr_t hFile = 0;   //win10
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		// "\\*"是指读取文件夹下的所有类型的文件，若想读取特定类型的文件，以png为例，则用“\\*.png”
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					folders.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}
void getFiles(string path, vector<string>& files)
{
	//文件句柄  
	intptr_t hFile = 0;   //win10

	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
		// "\\*"是指读取文件夹下的所有类型的文件，若想读取特定类型的文件，以png为例，则用“\\*.png”
	{
		do
		{
			if ((fileinfo.attrib & _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					continue;
			}
			else
			{
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

class BulletRender {
public:
	BulletRender(btVector3 gravity = btVector3(0, 0, 0)) {
		sim = new b3RobotSimulatorClientAPI_NoGUI();
		bool isConnected = sim->connect(eCONNECT_SHARED_MEMORY);
		if (!isConnected) {
			printf("Using Direct mode\n");
			isConnected = sim->connect(eCONNECT_DIRECT);
		}
		else {
			printf("Using shared memory\n");
		}
		sim->resetSimulation();
		sim->setGravity(gravity);
		sim->setNumSolverIterations(100);
	}
	~BulletRender() {
		delete sim;
	}

private:
	void AmassLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl);
	void AmassDofLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl);
	void PhyscapLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl);

public:
	void RenderMotion(int robotKind, CamMat mat, vector<string> posePath, vector<string> translPath, string robotPath, string terrainMeshPath = "");
	void RenderImgVideo(int robotKind, CamMat mat, const vector<string>& imgPaths, const string& savePath, const vector<string>& posePaths, const vector<string>& translPaths, const string& urdfPath);
	void RenderVideo(int robotKind, CamMat mat, const string& savePath, const vector<string>& posePaths, const vector<string>& translPaths, const string& urdfPath, const string terrainMeshPath = "");
	void RenderSmplVideo(CamMat mat, const string& savePath, const vector<string>& smplPath, const string terrainMeshPath = "");
	void RenderSmplImgVideo(CamMat mat, const vector<string>& imgPaths, const string& savePath, const vector<string>& smplPaths);

private:
	Config config;
	b3RobotSimulatorClientAPI_NoGUI* sim;
	Viewer_gui gui;

};

void BulletRender::AmassLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl) {
	if (!poses.size() || !transl.size()) {
		cout << "error: no poses or no transls data!" << endl;
		return;
	}
	for (auto key = 0; key < 24; key++) {
		int value = config.smplInAmassDof[key];
		if (key == 0) {
			Quaternionf q_out = AxisAngle2Quaternion(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			sim->resetBasePositionAndOrientation(
				bodyUid,
				btVector3(transl[0], transl[1], transl[2]),
				btQuaternion(q_out.coeffs()[0], q_out.coeffs()[1], q_out.coeffs()[2], q_out.coeffs()[3]));
		}
		else {
			Matrix3f R = AxisAngle2Rmat(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			vector<float> ordered_angles;
			vector<Eigen::Vector3f> xyz_order = { Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ() };
			Rmat2Euler(ordered_angles, R, xyz_order);
			b3JointInfo jointInfo;
			sim->getJointInfo(bodyUid, value, &jointInfo);
			for (auto rotkey = 0; rotkey < 3; rotkey++)
			{
				int dim = config.smplInAmassDofDim[key * 3 + rotkey];
				if (dim == 1)
				{
					sim->resetJointState(bodyUid, value + rotkey, ordered_angles[rotkey]);
				}
			}
		}
	}
}
void BulletRender::AmassDofLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl) {
	if (!poses.size() || !transl.size()) {
		cout << "error: no poses or no transls data!" << endl;
		return;
	}
	for (auto key = 0; key < 24; key++) {
		int value = config.smplInAmassOneDof[key];
		if (key == 0) {
			Quaternionf q_out = AxisAngle2Quaternion(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			sim->resetBasePositionAndOrientation(
				bodyUid,
				btVector3(transl[0], transl[1], transl[2]),
				btQuaternion(q_out.coeffs()[0], q_out.coeffs()[1], q_out.coeffs()[2], q_out.coeffs()[3]));
		}
		else if (value == -1) {
			continue;
		}
		else {
			Matrix3f R = AxisAngle2Rmat(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			vector<float> ordered_angles;
			vector<Eigen::Vector3f> xyz_order = { Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ() };
			Rmat2Euler(ordered_angles, R, xyz_order);
			b3JointInfo jointInfo;
			sim->getJointInfo(bodyUid, value, &jointInfo);
			for (auto rotkey = 0; rotkey < 3; rotkey++) {
				int dim = config.smplInAmssOneDofDim[key * 3 + rotkey];
				if (dim == 1) {
					sim->resetJointState(bodyUid, value + rotkey, ordered_angles[rotkey]);
				}
			}
		}
	}
}
void BulletRender::PhyscapLoadMotion(int& bodyUid, VectorXf& poses, Vector3f& transl) {
	if (!poses.size() || !transl.size()) {
		cout << "error: no poses or no transls data!" << endl;
		return;
	}
	for (auto key = 0; key < 24; key++) {
		int value = config.smplInPhyscap[key];
		if (key == 0) {
			Quaternionf q_out = AxisAngle2Quaternion(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			sim->resetBasePositionAndOrientation(
				bodyUid,
				btVector3(transl[0], transl[1], transl[2]),
				btQuaternion(q_out.coeffs()[0], q_out.coeffs()[1], q_out.coeffs()[2], q_out.coeffs()[3]));
		}
		else {
			Matrix3f R = AxisAngle2Rmat(Vector3f(poses[key * 3], poses[key * 3 + 1], poses[key * 3 + 2]));
			vector<float> ordered_angles;
			vector<Eigen::Vector3f> xyz_order = { Eigen::Vector3f::UnitX(), Eigen::Vector3f::UnitY(), Eigen::Vector3f::UnitZ() };
			Rmat2Euler(ordered_angles, R, xyz_order);
			b3JointInfo jointInfo;
			sim->getJointInfo(bodyUid, value, &jointInfo);
			for (auto rotkey = 0; rotkey < 3; rotkey++) {
				int dim = config.smplInPhyscapDofDim[key * 3 + rotkey];
				if (dim == 1) {
					sim->resetJointState(bodyUid, value + rotkey, ordered_angles[rotkey]);
				}
			}
		}
	}
}

void BulletRender::RenderMotion(int robotKind, CamMat mat, vector<string> posePath, vector<string> translPath, string robotPath, string terrainMeshPath) {
	int frameNum = posePath.size();
	vector<VectorXf> poses;
	vector<Vector3f> transl;
	for (auto i = 0; i < frameNum; i++) {
		MatrixXf posem, translm;
		load_nptxt(posem, posePath[i]);
		load_nptxt(translm, translPath[i]);
		poses.push_back(posem.col(0));
		transl.push_back(translm.col(0));
	}

	vector<int> bodyUids;
	for (auto i = 0; i < frameNum; i++) bodyUids.push_back(sim->loadURDF(robotPath));
	gui.active_shadow(true);
	render_id floor_id = gui.add_floor(50, Vector3f(0, 0, 0), Vector3f(0, 1, 0),
		Vector3f(120.0 / 255, 120.0 / 255, 120.0 / 255), Vector3f(0.99, 0.99, 0.99));

	BtRobotSimProxy phyproxy;
	if (terrainMeshPath != "") {
		MatrixXf Boxes;
		load_nptxt(Boxes, terrainMeshPath);
		phyproxy.addBoxTerrain(gui, Boxes);
	}
	for (auto i = 0; i < frameNum; i++) {
		phyproxy.addBindingMultiBody(gui, sim, bodyUids[i]);
		if (robotKind == Physcap)
			PhyscapLoadMotion(bodyUids[i], poses[i], transl[i]);
		if (robotKind == Amass)
			AmassLoadMotion(bodyUids[i], poses[i], transl[i]);
		if (robotKind == AmassDof)
			AmassDofLoadMotion(bodyUids[i], poses[i], transl[i]);
	}

	if (!mat.inmat.isZero()) {
		if (mat.exmat.isZero()) {
			mat.exmat.setIdentity();
			AngleAxisf AA(mat.v.norm(), mat.v / mat.v.norm());
			mat.exmat.block(0, 0, 3, 3) = AA.toRotationMatrix();
			mat.exmat.block(0, 3, 3, 1) = mat.trans;
		}
		gui.step_launch(mat.windowSize, { mat.inmat }, { mat.exmat });
	}
	else {
		gui.step_launch(mat.windowSize);
	}

	vector<int>render_ids;
	vector<MatrixXf> render_verts;
	phyproxy.stepBindingUpdate(render_ids, render_verts, sim, bodyUids);
	while (!gui.step_close_window()) {
		if (gui.step_pause()) {
			phyproxy.stepBindingUpdate(render_ids, render_verts, sim, bodyUids);
		}
		gui.step_updateVerts(render_ids, render_verts);
		gui.step_refresh();
	}
	gui.step_terminate();
}
void BulletRender::RenderVideo(int robotKind, CamMat mat, const string& savePath, const vector<string>& posePaths, const vector<string>& translPaths, const string& urdfPath, const string terrainMeshPath) {
	gui.active_shadow(true);
	render_id floor_id = gui.add_floor(50, Vector3f(0, 0, 0), Vector3f(0, 1, 0),
		Vector3f(120.0 / 255, 120.0 / 255, 120.0 / 255), Vector3f(0.99, 0.99, 0.99));

	int frameNum = posePaths.size();
	vector<VectorXf> poses;
	vector<Vector3f> transl;
	for (auto i = 0; i < frameNum; i++) {
		MatrixXf posem, translm;
		load_nptxt(posem, posePaths[i]);
		load_nptxt(translm, translPaths[i]);
		poses.push_back(posem.col(0));
		transl.push_back(translm.col(0));
	}

	int bodyUid = sim->loadURDF(urdfPath);
	BtRobotSimProxy phyproxy;
	phyproxy.addBindingMultiBody(gui, sim, bodyUid);
	if (terrainMeshPath != "") {
		MatrixXf Boxes;
		load_nptxt(Boxes, terrainMeshPath);
		phyproxy.addBoxTerrain(gui, Boxes);
	}

	if (!mat.inmat.isZero()) {
		if (mat.exmat.isZero()) {
			mat.exmat.setIdentity();
			AngleAxisf AA(mat.v.norm(), mat.v / mat.v.norm());
			mat.exmat.block(0, 0, 3, 3) = AA.toRotationMatrix();
			mat.exmat.block(0, 3, 3, 1) = mat.trans;
		}
		gui.step_launch(mat.windowSize, { mat.inmat }, { mat.exmat });
	}
	else {
		gui.step_launch(mat.windowSize);
	}

	vector<int>render_ids; // {3, 2, 1}
	vector<MatrixXf> render_verts;

	if (robotKind == Physcap) PhyscapLoadMotion(bodyUid, poses[0], transl[0]);
	if (robotKind == Amass) AmassLoadMotion(bodyUid, poses[0], transl[0]);
	if (robotKind == AmassDof) AmassDofLoadMotion(bodyUid, poses[0], transl[0]);

	phyproxy.stepBindingUpdate(render_ids, render_verts, sim, { bodyUid });

	gui.step_updateVerts(render_ids, render_verts);
	gui.step_refresh();

	int idx = 0;
	while (!gui.step_close_window()) {
		if (gui.step_pause()) {
			if (robotKind == Physcap) PhyscapLoadMotion(bodyUid, poses[idx], transl[idx]);
			if (robotKind == Amass) AmassLoadMotion(bodyUid, poses[idx], transl[idx]);
			if (robotKind == AmassDof) AmassDofLoadMotion(bodyUid, poses[idx], transl[idx]);

			phyproxy.stepBindingUpdate(render_ids, render_verts, sim, { bodyUid });

			gui.step_updateVerts(render_ids, render_verts);
			gui.step_refresh();

			WinPara* ptr = (WinPara*)glfwGetWindowUserPointer(gui.m_windows[0].ptrWindow);
			Mat image(ptr->height, ptr->width, CV_8UC3);
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glPixelStorei(GL_PACK_ROW_LENGTH, ptr->width);
			glReadPixels(0, 0, ptr->width, ptr->height, GL_BGR, GL_UNSIGNED_BYTE, image.data);
			flip(image, image, 0);
			imwrite(savePath + zfill(to_string(idx), 5) + ".png", image);

			idx += 1;
			if (idx >= frameNum) break;
		}
		gui.step_updateVerts(render_ids, render_verts);
		gui.step_refresh();
	}
	gui.step_terminate();
}

void BulletRender::RenderImgVideo(int robotKind, CamMat mat, const vector<string>& imgPaths, const string& savePath, const vector<string>& posePaths, const vector<string>& translPaths, const string& urdfPath) {
	int num = posePaths.size();
	int bodyUid = sim->loadURDF(urdfPath);

	MatrixXf dif;
	load_nptxt(dif, "H:\\YangYuan\\Code\\phy_program\\CodeBase\\dif.txt");

	for (auto i = 0; i < num; i++) {
		BtRobotSimProxy phyproxy;
		phyproxy.addBindingMultiBody(gui, sim, bodyUid);
		MatrixXf posem, translm;
		load_nptxt(posem, posePaths[i]);
		load_nptxt(translm, translPaths[i]);
		VectorXf pose = posem.col(0);
		Vector3f transl = translm.col(0);

		if (robotKind == Physcap) PhyscapLoadMotion(bodyUid, pose, transl);
		if (robotKind == Amass) AmassLoadMotion(bodyUid, pose, transl);
		if (robotKind == AmassDof) AmassDofLoadMotion(bodyUid, pose, transl);
		Mat img = imread(imgPaths[i]);
		mat.windowSize << img.cols, img.rows;

		Matrix4f exmat = mat.exmat;
		exmat(0, 3) -= dif(i, 0);
		exmat(1, 3) -= dif(i, 1);
		exmat(2, 3) -= dif(i, 2);

		vector<Mat> background_img_list = { img };
		gui.add_background({ imread(imgPaths[i]) });
		gui.step_launch(Vector2i(img.cols, img.rows), { mat.inmat }, { exmat });
		vector<int>render_ids;
		vector<MatrixXf> render_verts;
		phyproxy.stepBindingUpdate(render_ids, render_verts, sim, { bodyUid });
		gui.step_updateVerts(render_ids, render_verts);
		gui.step_refresh();

		WinPara* ptr = (WinPara*)glfwGetWindowUserPointer(gui.m_windows[0].ptrWindow);
		Mat image(ptr->height, ptr->width, CV_8UC3);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ROW_LENGTH, ptr->width);
		glReadPixels(0, 0, ptr->width, ptr->height, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		flip(image, image, 0);
		imwrite(savePath + zfill(to_string(i), 10) + ".png", image);

		gui.step_terminate();
	}
}

void BulletRender::RenderSmplVideo(CamMat mat, const string& savePath, const vector<string>& smplPaths, const string terrainMeshPath) {
	gui.active_shadow(true);
	render_id floor_id = gui.add_floor(50, Vector3f(0, 0, 0), Vector3f(0, 1, 0),
		Vector3f(120.0 / 255, 120.0 / 255, 120.0 / 255), Vector3f(0.99, 0.99, 0.99));

	int frameNum = smplPaths.size();

	BtRobotSimProxy phyproxy;
	if (terrainMeshPath != "") {
		MatrixXf Boxes;
		load_nptxt(Boxes, terrainMeshPath);
		phyproxy.addBoxTerrain(gui, Boxes);
	}

	MatrixXf vs, ts;
	MatrixXi fs;
	load_elementary_mesh(smplPaths[0], vs, fs, ts);
	gui.add_mesh(vs, fs);
	if (!mat.inmat.isZero()) {
		if (mat.exmat.isZero()) {
			mat.exmat.setIdentity();
			AngleAxisf AA(mat.v.norm(), mat.v / mat.v.norm());
			mat.exmat.block(0, 0, 3, 3) = AA.toRotationMatrix();
			mat.exmat.block(0, 3, 3, 1) = mat.trans;
		}
		gui.step_launch(mat.windowSize, { mat.inmat }, { mat.exmat });
	}
	else {
		gui.step_launch(mat.windowSize);
	}

	gui.step_refresh();

	while (!gui.step_close_window()) {
		if (gui.step_pause()) {
			vector<Eigen::Matrix3f> camera_Intrins;
			vector<Eigen::Matrix4f> camera_Extrins;
			gui.computeIntrinsicfromProjection(camera_Intrins);
			gui.computeExtrinsicfromModelview(camera_Extrins);
			printf("\n[GUI]Intrins (%d) : \n", int(camera_Intrins.size()));
			cout << camera_Intrins[0] << endl;

			mat.inmat = camera_Intrins[0];

			printf("[GUI]Extrins (%d) : \n", int(camera_Extrins.size()));
			Eigen::Matrix3f camera_mat = camera_Extrins[0].topLeftCorner(3, 3);
			AngleAxisf rot = AngleAxisf(camera_mat);
			cout << "\t Rot(axis -- angle): " << rot.axis().transpose()
				<< "--" << rot.angle() << endl;
			cout << "\t trans: " << camera_Extrins[0].topRightCorner(3, 1).transpose() << endl;

			mat.v = rot.axis() * rot.angle();
			mat.trans = camera_Extrins[0].topRightCorner(3, 1);
			break;
		}
		gui.step_refresh();
	}

	while (!gui.step_close_window()) {
		if (gui.step_pause()) {
			for (auto i = 0; i < frameNum; i++) {
				BtRobotSimProxy phyproxy;
				MatrixXf vs, ts;
				MatrixXi fs;
				load_elementary_mesh(smplPaths[i], vs, fs, ts);
				render_id floor_id = gui.add_floor(50, Vector3f(0, 0, 0), Vector3f(0, 1, 0),
					Vector3f(120.0 / 255, 120.0 / 255, 120.0 / 255), Vector3f(0.99, 0.99, 0.99));
				gui.add_mesh(vs, fs);
				if (!mat.inmat.isZero()) {
					if (mat.exmat.isZero()) {
						mat.exmat.setIdentity();
						AngleAxisf AA(mat.v.norm(), mat.v / mat.v.norm());
						mat.exmat.block(0, 0, 3, 3) = AA.toRotationMatrix();
						mat.exmat.block(0, 3, 3, 1) = mat.trans;
					}
					gui.step_launch(mat.windowSize, { mat.inmat }, { mat.exmat });
				}
				else {
					gui.step_launch(mat.windowSize);
				}
				gui.step_refresh();

				WinPara* ptr = (WinPara*)glfwGetWindowUserPointer(gui.m_windows[0].ptrWindow);
				Mat image(ptr->height, ptr->width, CV_8UC3);
				glPixelStorei(GL_PACK_ALIGNMENT, 1);
				glPixelStorei(GL_PACK_ROW_LENGTH, ptr->width);
				glReadPixels(0, 0, ptr->width, ptr->height, GL_BGR, GL_UNSIGNED_BYTE, image.data);
				flip(image, image, 0);
				imwrite(savePath + zfill(to_string(i), 6) + ".png", image);

				gui.step_terminate();
			}
			return;
		}
		gui.step_refresh();
	}
}

void BulletRender::RenderSmplImgVideo(CamMat mat, const vector<string>& imgPaths, const string& savePath, const vector<string>& smplPaths)
{
	int num = smplPaths.size();

	MatrixXf dif;
	load_nptxt(dif, "H:\\YangYuan\\Code\\phy_program\\CodeBase\\dif.txt");

	for (auto i = 0; i < num; i++) {
		MatrixXf vs, ts;
		MatrixXi fs;
		load_elementary_mesh(smplPaths[i], vs, fs, ts);
		gui.add_mesh(vs, fs);
		Mat img = imread(imgPaths[i]);
		mat.windowSize << img.cols, img.rows;

		Matrix4f exmat = mat.exmat;
		//exmat(0, 3) -= dif(i, 0);
		//exmat(1, 3) -= dif(i, 1);
		//exmat(2, 3) -= dif(i, 2);

		vector<Mat> background_img_list = { img };
		gui.add_background({ imread(imgPaths[i]) });
		gui.step_launch(Vector2i(img.cols, img.rows), { mat.inmat }, { exmat });
		vector<int>render_ids;
		vector<MatrixXf> render_verts;
		gui.step_refresh();

		WinPara* ptr = (WinPara*)glfwGetWindowUserPointer(gui.m_windows[0].ptrWindow);
		Mat image(ptr->height, ptr->width, CV_8UC3);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glPixelStorei(GL_PACK_ROW_LENGTH, ptr->width);
		glReadPixels(0, 0, ptr->width, ptr->height, GL_BGR, GL_UNSIGNED_BYTE, image.data);
		flip(image, image, 0);
		imwrite(savePath + zfill(to_string(i), 10) + ".png", image);

		gui.step_terminate();
	}
}

int main()
{
	// RenderMotion Test
	/* {
		BulletRender renderTest;
		CamMat mat;

		string rootPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\";
		string terrainMesh = "0034_box.txt";
		string robot = "shape-0034-000.urdf";
		vector<string> posepath;
		vector<string> transpath;
		posepath.push_back(rootPath + "motionData\\supmat\\" + "GPA0034_Camera04_00297_ours" + "_pose.txt");
		transpath.push_back(rootPath + "motionData\\supmat\\" + "GPA0034_Camera04_00297_ours" + "_transl.txt");

		// Given camera angle
		//mat.inmat <<
		//	1854.12, 0, 1024,
		//	0, 1854.12, 768,
		//	0, 0, 1;
		//mat.v << -0.995714, -0.0227265, -0.0896512;
		//mat.v *= 3.08169;
		//mat.trans << 0.488401, 1.31906, 7.92661;

		renderTest.RenderMotion(
			AmassDof, mat,
			posepath,
			transpath,
			rootPath + "urdf\\" + robot,
			rootPath + "urdf\\" + terrainMesh);
	}*/

	// RenderVideo Test
	/* {
		BulletRender renderTest;
		CamMat mat;
		mat.windowSize << 960, 960;
		mat.inmat <<
			1158.82, 0, 480,
			0, 1158.82, 480,
			0, 0, 1;
		mat.v << -0.700215, -0.0170787, 0.713728;
		mat.v *= 3.13796;
		mat.trans << -0.174408, 0.799853, 4.81257;

		string rootPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\new_smooth\\data\\";
		string urdfPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\demo1Dof.urdf";
		string savePath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\new_smooth\\render\\";

		vector<string> posePaths, translPaths;

		for (auto i = 0; i <= 285; i++){
			posePaths.push_back(rootPath + zfill(to_string(i), 6) + "_pose.txt");
			translPaths.push_back(rootPath + zfill(to_string(i), 6) + "_transl.txt");
		}

		renderTest.RenderVideo(AmassDof, mat, savePath, posePaths, translPaths, urdfPath);
	}*/

	// RenderImgVideo Test
	/* {
		BulletRender renderTest;
		CamMat mat;

		string imgPath = "\\\\105.1.1.112\\e\\Human-Data-Physics-v1.0\\kinematic-huawei\\images\\demo1\\Camera00";
		string rootPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\ref_offset_adj_spline_fitting_sample\\data";
		string urdfPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\demo1Dof.urdf";
		string camPath = "\\\\105.1.1.112\e\Human-Data-Physics-v1.0\kinematic-huawei\camparams\demo1\\camparams.txt";
		string savePath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\ref_offset_adj_spline_fitting_sample\\renderImg\\";
		vector<Matrix3f> camIns;
		vector<Matrix4f> camExs;
		load_cam_params(camPath, camIns, camExs);

		// mat.inmat = camIns[0]; mat.exmat = camExs[0];
		mat.inmat <<
			2000, 0, 184,
			0, 2000, 320,
			0, 0, 1;
		mat.exmat <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;

		Matrix4f exoff = Matrix4f::Zero();
		exoff <<
			-0.65130947, -0.0044586, -0.75879911, 32.31011,
			0.0044586, -0.99998796, 0.00204878, -1.8454262,
			-0.75879911, -0.00204878, 0.65132151, -3.9115276,
			0, 0, 0, 1;
		mat.exmat <<
			-0.65130947, 0.00445859, -0.75879911, 18.08404511,// - 16.80981565,
			-0.00445859, -0.99998796, -0.00204879, -1.70936021,// + 1.34898949,
			-0.75879911, 0.00204879, 0.65132151, 27.06832579,// - 7.72473258,
			0, 0, 0, 1;

		vector<string> imgPaths, pklPaths, posePaths, translPaths;
		getFiles(imgPath, imgPaths);
		for (auto& imgpath : imgPaths)
		{
			imgpath = imgPath + "\\" + imgpath;
		}
		for (auto i = 0; i <= 285;i++)
		{
			posePaths.push_back(rootPath + "\\" + zfill(to_string(i), 6) + "_pose.txt");
			translPaths.push_back(rootPath + "\\" + zfill(to_string(i), 6) + "_transl.txt");
		}
		renderTest.RenderImgVideo(AmassDof, mat, imgPaths, savePath, posePaths, translPaths, urdfPath);
	}*/

	// RenderSmplVideo Test
	/* {
		BulletRender renderTest;
		CamMat mat;
		//mat.inmat <<
		//	1854.12, 0, 1024,
		//	0, 1854.12, 768,
		//	0, 0, 1;
		//mat.v << 0.499894,-0.0836677,-0.862036;
		//mat.v *= 3.05159;
		//mat.trans << -0.843589,2.06126,6.96592;
		//mat.inmat <<
		//	1854.12, 0, 1024,
		//	0, 1854.12, 768,
		//	0, 0, 1;
		//mat.v << 1, 0, 0;
		//mat.v *= 3.14159;
		//mat.trans << -0.264773,0.776739,7.33657;
		//mat.inmat <<
		//	1854.12, 0, 1024,
		//	0, 1854.12, 768,
		//	0, 0, 1;
		//mat.v << 0.924665,0.0204612,0.38023;
		//mat.v *= 3.10213;
		//mat.trans << 0.27038,1.25352,5.11614;
		string rootPath = "H:\\YangYuan\\Code\\phy_program\\CodeBase\\data\\temdata\\results\\huaweiMesh\\params";
		string savePath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\renderImg\\test\\";

		vector<string> smplPaths;
		for (auto i = 0; i < 1452; i++) {
			smplPaths.push_back(rootPath + "\\" + zfill(to_string(i),6)+".obj");
		}

		renderTest.RenderSmplVideo(mat, savePath, smplPaths);

	}*/

	// RenderSmplVideo Test
	{
		BulletRender renderTest;
		CamMat mat;

		string imgPath = "\\\\105.1.1.112\\e\\Human-Data-Physics-v1.0\\kinematic-huawei\\images\\demo1\\Camera00";
		string rootPath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\new_smooth\\smpl";
		string camPath = "\\\\105.1.1.112\e\Human-Data-Physics-v1.0\kinematic-huawei\camparams\demo1\\camparams.txt";
		string savePath = "H:\\YangYuan\\Code\\cpp_program\\seuvcl-codebase-master2\\data\\graphics\\physdata\\motionData\\output\\new_smooth\\renderSmpl\\";
		vector<Matrix3f> camIns;
		vector<Matrix4f> camExs;
		load_cam_params(camPath, camIns, camExs);

		// mat.inmat = camIns[0]; mat.exmat = camExs[0];
		mat.inmat <<
			2000, 0, 184,
			0, 2000, 320,
			0, 0, 1;
		mat.exmat <<
			-0.65130947, 0.00445859, -0.75879911, 18.08404511,// - 16.80981565,
			-0.00445859, -0.99998796, -0.00204879, -1.70936021,// + 1.34898949,
			-0.75879911, 0.00204879, 0.65132151, 27.06832579,// - 7.72473258,
			0, 0, 0, 1;
		mat.exmat <<
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1;
		//mat.exmat <<
		//	-0.65130947, 0.00445859, -0.75879911, 1.27502768,
		//	-0.00445859, -0.99998796, -0.00204879, -0.20132872,
		//	-0.75879911, 0.00204879, 0.65132151, 19.34436249,
		//	0, 0, 0, 1;
		//mat.exmat <<
		//	0.86072595, 0.01294855, 0.50890389, 1.6296913,
		//	-0.01294855, -0.99879615, 0.04731368, -0.32722951,
		//	0.50890389, -0.04731368, -0.85952211, 24.40299535,
		//	0, 0, 0, 1;
		//mat.exmat <<
		//	0.95575415, -0.03807697, 0.29169185, -0.77602318,
		//	0.03807697, -0.96723183, -0.25102339, 1.84465294,
		//	0.29169185,  0.25102339, -0.92298598, 2.65906859,
		//	0, 0, 0, 1;

		vector<string> imgPaths, pklPaths, smplPaths;
		getFiles(imgPath, imgPaths);
		for (auto& imgpath : imgPaths)
		{
			imgpath = imgPath + "\\" + imgpath;
		}
		for (auto i = 0; i <= 286; i++)
		{
			smplPaths.push_back(rootPath + "\\" + zfill(to_string(i), 6) + ".obj");
		}
		renderTest.RenderSmplImgVideo(mat, imgPaths, savePath, smplPaths);

	}
}