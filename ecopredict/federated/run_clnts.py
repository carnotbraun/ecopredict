import subprocess
import time
from multiprocessing import Process

def start_client(client_id):
    process = subprocess.Popen(
        ["python3", "client.py", str(client_id)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    print(stdout.decode())
    if stderr:
        print(stderr.decode())

def run_clients(num_clients):
    processes = []
    for client_id in range(num_clients):
        process = Process(target=start_client, args=(client_id,))
        process.start()
        processes.append(process)
        print(f"Cliente {client_id} iniciado")
    
    # Opcional: aguarde alguns segundos para garantir que todos os clientes estejam totalmente iniciados
    time.sleep(5)
    print("Todos os clientes foram iniciados.")

    # Aguarda todos os processos terminarem
    for process in processes:
        process.join()

if __name__ == "__main__":
    num_clients = 6  # Defina o número de clientes que deseja executar
    run_clients(num_clients)
