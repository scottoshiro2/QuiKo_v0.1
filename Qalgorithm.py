from math import *
import numpy as np
import matplotlib.pyplot as plt

from qiskit import *
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute
from qiskit.circuit import Parameter
from qiskit.tools.visualization import plot_histogram

from qiskit import Aer, IBMQ
from qiskit.providers.ibmq import least_busy
from qiskit.tools.monitor import job_monitor

from mapping import Qstate_dict

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def LQR(Qcircuit, n_qubits=4):
    LQR = Qcircuit
    for i in np.arange(n_qubits):
        LQR.h(i)
        
    return LQR 

def SpinalQ( Qcircuit, n_qubits, XROT_theta, ZROT_psi): ## n_qubits should be length of Rx
    num_tats = len(ZROT_psi)

    theta = Parameter('theta') ## Note: Need to adjust the SigPro to get freq of num of tatums present
    psi   = Parameter('psi')
    gamma = Parameter('gam')
    
    spine = np.arange(0, n_qubits)
    spine = spine[spine > 3]
    
    spine_circuit = Qcircuit
    spine_circuit.barrier()
    
    for index, Ugate in enumerate(zip(XROT_theta,ZROT_psi)): ## Rx is the number of tatum onsets in the current beat
        if index <= 3:
            spine_circuit.cu3(theta, psi, gamma, index, spine[index])
            
        else:
            icyc = index - 4
            spine_circuit.cu3(theta, psi, gamma, icyc, spine[icyc])
            
        spine_circuit = spine_circuit.bind_parameters({theta: round(Ugate[0], 3)})
        spine_circuit = spine_circuit.bind_parameters({psi: round(Ugate[1], 3)})
        spine_circuit = spine_circuit.bind_parameters({gamma: 0})
    
    spine_circuit.barrier()
    
    return spine_circuit, num_tats

def qft_dagger(circ, n): ## Make sure to site this function from the qiskit textbook
    """n-qubit QFTdagger the first n qubits in circ"""
    for j in range(n):
        for m in range(j):
            circ.cu1(-np.pi/float(2**(j-m)), m, j)
        circ.h(j)
        
    return circ

def build_circuit(n_qubits, n_bits, Ry_Base):
    
    circ_array = [] 

    for index, circuit in enumerate(Ry_Base):
        Rx = Ry_Base[f'Beat{index+1}'][f'Rx {index+1}']
        Rz = Ry_Base[f'Beat{index+1}'][f'Rz {index+1}']
        
        ZROT_psi   = (Rz / 1000) * 360 #This is the phase rotation
        XROT_theta = Rx * 360 #This is the drive amplitude
        
        tqr  = QuantumRegister(n_qubits)
        tcr1 = ClassicalRegister(n_bits)
        tcr2 = ClassicalRegister(n_bits)

        Qtest_full = QuantumCircuit( tqr, tcr1, tcr2 )

        phase_est = LQR(Qtest_full)
        Spine, num_tats = SpinalQ(phase_est, n_qubits, XROT_theta, ZROT_psi)
        QuiKo = qft_dagger( Spine, 4); QuiKo.barrier()
        
        ## Measurement Stage:

        QuiKo.measure( tqr[:4],tcr1[:4] )
        QuiKo.measure( tqr[4:],tcr2[:4] )
                
        circ_array.append([QuiKo, num_tats])
        
    return circ_array

##~~~~~Running simulator and Quanutm devices~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def simulator_backend(circ_array):
    backend = Aer.get_backend("qasm_simulator")
    res_array = []

    for index, Qcirc in enumerate(circ_array):
        simulate = execute(Qcirc, backend=backend, shots=1024).result()
        res_array.append(simulate.get_counts()) 
    
    return res_array

def IBMQ_backend(circ_array):
    IBMQ.load_account()
    provider = IBMQ.get_provider('ibm-q')
    backend = provider.get_backend('ibmq_16_melbourne')
    res_array = []

    for index, Qcirc in enumerate(circ_array):
        job = execute(Qcirc, backend=backend, shots=1024).result()
        res_array.append(job.get_counts()) 
    
    return res_array 

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def prob_dist(result_dict):

	prob_dist = []; pd = []
	states = []; st = []
	for index, result in enumerate(result_dict):
		for element in result:
			pd.append(result[element])
			st.append(element)

		prob_dist.append(pd); pd = []
		states.append(st); st = []

	return prob_dist, states

def QuiKo_Algorithm(audiosample, simulator=True): 
    filename = 'Samples/'+audiosample+'.wav'
    Ry_Base, tempo = Qstate_dict(filename)
    #IBMQ.load_account()
    n_qubits = 8; n_bits = 4 
    
    circ_array = build_circuit(n_qubits, n_bits, Ry_Base)
    circ_array[1][0].draw(output="mpl", filename='MQC2')

    num_tats   = [circ_array[i][1] for i, val in enumerate(circ_array)]
    circ_array = [circ_array[i][0] for i, val in enumerate(circ_array)]

    if simulator == True: ## Toss it to the backend
        results = simulator_backend(circ_array)
    else:
        results = IBMQ_backend(circ_array)


    for index, result in enumerate(results):
        total_counts = sum(results[index].values())

        for key in result:
            result[key] = result[key] / total_counts

    return results, num_tats, tempo



