from xml.etree.ElementTree import Element, SubElement, tostring
import xml.etree.ElementTree as ET
import xml.dom.minidom
from transforms3d.quaternions import mat2quat
from transforms3d.euler import quat2euler
import numpy as np

class xmlReader():
    def __init__(self, xmlfilename):
        self.xmlfilename = xmlfilename
        etree = ET.parse(self.xmlfilename)
        self.top = etree.getroot()

    def showinfo(self):
        print('Resumed object(s) already stored in '+self.xmlfilename+':')
        for i in range(len(self.top)):
            print(self.top[i][1].text)

    def gettop(self):
        return self.top

    def getposevectorlist(self):
        # posevector foramat: [objectid,x,y,z,alpha,beta,gamma]
        posevectorlist = []
        for i in range(len(self.top)):
            objectid = int(self.top[i][0].text)
            objectname = self.top[i][1].text
            objectpath = self.top[i][2].text
            translationtext = self.top[i][3].text.split()
            translation = []
            for text in translationtext:
                translation.append(float(text))
            quattext = self.top[i][4].text.split()
            quat = []
            for text in quattext:
                quat.append(float(text))
            alpha, beta, gamma = quat2euler(quat)
            x, y, z = translation
            alpha *= (180.0 / np.pi)
            beta *= (180.0 / np.pi)
            gamma *= (180.0 / np.pi)
            posevectorlist.append([objectid, x, y, z, alpha, beta, gamma])
        return posevectorlist

    def get_pose_list(self):
        pose_vector_list = self.getposevectorlist()
        return pose_list_from_pose_vector_list(pose_vector_list)



from transforms3d.euler import euler2quat
class Pose:
    def __init__(self,id,x,y,z,alpha,beta,gamma):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        # alpha, bata, gamma is in degree
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quat = self.get_quat()
        self.mat_4x4 = self.get_mat_4x4()
        self.translation = self.get_translation()

    def __repr__(self):
        return '\nPose id=%d,x=%f,y=%f,z=%f,alpha=%f,beta=%f,gamma=%f' %(self.id,self.x,self.y,self.z,self.alpha,self.beta,self.gamma)+'\n'+'translation:'+self.translation.__repr__() + '\nquat:'+self.quat.__repr__()+'\nmat_4x4:'+self.mat_4x4.__repr__()

    def get_id(self):
        """
        Function:
        return the id of this object
        """
        return self.id

    def get_translation(self):
        """ 
        Function:
        Convert self.x, self.y, self.z into self.translation
        """
        return np.array([self.x,self.y,self.z])

    def get_quat(self):
        """
        Function:
        Convert self.alpha, self.beta, self.gamma into self.quat
        """
        euler = np.array([self.alpha, self.beta, self.gamma]) / 180.0 * np.pi
        quat = euler2quat(euler[0],euler[1],euler[2])
        return quat

    def get_mat_4x4(self):
        """
        Function:
        Convert self.x, self.y, self.z, self.alpha, self.beta and self.gamma into mat_4x4 pose
        """
        mat_4x4 = trans3d.get_mat(self.x,self.y,self.z,self.alpha,self.beta,self.gamma)
        return mat_4x4

def pose_from_pose_vector(pose_vector):
    """
    Input:
    pose_vector: A list in the format of [id,x,y,z,alpha,beta,gamma]
    
    Output:
    A pose class instance
    """
    return Pose(id = pose_vector[0],
    x = pose_vector[1],
    y = pose_vector[2],
    z = pose_vector[3],
    alpha = pose_vector[4],
    beta = pose_vector[5],
    gamma = pose_vector[6])

def pose_list_from_pose_vector_list(pose_vector_list):
    """
    Input:
    Pose vector list defined in xmlhandler.py

    Output:
    A list of poses.
    """
    pose_list = []
    for pose_vector in pose_vector_list:
        pose_list.append(pose_from_pose_vector(pose_vector))
    return pose_list


from transforms3d.quaternions import mat2quat, quat2mat
from transforms3d.euler import quat2euler, euler2quat
import numpy as np

def get_pose(pose):
	pos, quat = pose_4x4_to_pos_quat(pose)
	euler = np.array([quat2euler(quat)[0], quat2euler(quat)[1],quat2euler(quat)[2]])
	euler = euler * 180.0 / np.pi
	alpha, beta, gamma = euler[0], euler[1], euler[2]
	x, y, z = pos[0], pos[1], pos[2]
	return x,y,z, alpha, beta, gamma

def get_mat(x,y,z, alpha, beta, gamma):
	"""
	Calls get_mat() to get the 4x4 matrix
	"""
	try:
		euler = np.array([alpha, beta, gamma]) / 180.0 * np.pi
		quat = np.array(euler2quat(euler[0],euler[1],euler[2]))
		pose = pos_quat_to_pose_4x4(np.array([x,y,z]), quat)
		return pose
	except Exception as e:
		print(str(e))
		pass         

def pos_quat_to_pose_4x4(pos, quat):
	"""pose = pos_quat_to_pose_4x4(pos, quat)
	Convert pos and quat into pose, 4x4 format

	Args:
	    pos: length-3 position
	    quat: length-4 quaternion

	Returns:
	    pose: numpy array, 4x4
	"""
	pose = np.zeros([4, 4])
	mat = quat2mat(quat)
	pose[0:3, 0:3] = mat[:, :]
	pose[0:3, -1] = pos[:]
	pose[-1, -1] = 1
	return pose


def pose_4x4_to_pos_quat(pose):
	"""
	Convert pose, 4x4 format into pos and quat

	Args:
	    pose: numpy array, 4x4
	Returns:
		pos: length-3 position
	    quat: length-4 quaternion

	"""
	mat = pose[:3, :3]
	quat = mat2quat(mat)
	pos = np.zeros([3])
	pos[0] = pose[0, 3]
	pos[1] = pose[1, 3]
	pos[2] = pose[2, 3]
	return pos, quat