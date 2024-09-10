import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt

# Lista para armazenar as perdas de cada rodada
losses = []
errors = []
times = []
# Estratégia personalizada para salvar pesos e avaliar as perdas
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results: list,
        failures: list[BaseException],
    ) -> tuple[float, dict]:
        print(f"Round {rnd}: results={results}, failures={failures}")  # Adiciona instrução de depuração
        
        loss, evaluate_metrics = super().aggregate_evaluate(rnd, results, failures)
        losses.append(loss)
        for res in results:
            # Debug print para verificar a estrutura do resultado
            print(f"Result structure: {res}")
            
            client_proxy, evaluate_res = res
            if isinstance(evaluate_res, fl.server.client_proxy.EvaluateRes):
                client_metrics = evaluate_res.metrics
                if "rsu_id" in client_metrics:
                    rsu_id = client_metrics["rsu_id"]
                    mse = evaluate_res.loss
                    errors.append([rsu_id, mse, rnd])
        return loss, evaluate_metrics
    
    def aggregate_fit(
        self,
        rnd: int,
        results: list,
        failures: list[BaseException],
    ) -> tuple[float, dict]:
        print(f"Round {rnd}: results={results}, failures={failures}")
                
        for res in results:
            # Debug print para verificar a estrutura do resultado
            print(f"Result structure: {res}")
            times.append([res[1].metrics["rsu_id"], res[1].metrics["time"],  rnd])

        return super().aggregate_fit(rnd, results, failures)
        
    
# Função para salvar o time de treinamento em um arquivo CSV
def save_times_to_csv(times, filename="times.csv"):
    df = pd.DataFrame(times, columns=['RSU', 'TIME', 'ROUND'])
    df.to_csv(filename, index=False)
    
# Função para salvar perdas em um arquivo CSV
def save_losses_to_csv(losses, filename="losses.csv"):
    df = pd.DataFrame({"round": range(1, len(losses) + 1), "loss": losses})
    df.to_csv(filename, index=False)

# Função para salvar erros por RSU em um arquivo CSV
def save_errors_to_csv(errors, filename="lstm_errors.csv"):
    df = pd.DataFrame(errors, columns=['RSU', 'MSE', 'ROUND'])
    df.to_csv(filename, index=False)

# Iniciar o servidor federado
if __name__ == "__main__":
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=6,
        min_evaluate_clients=6,
        min_available_clients=6,
    )
    
    fl.server.start_server(
        server_address="[::]:9999",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

    # Salvar as perdas em um arquivo CSV
    save_times_to_csv(times)
    save_errors_to_csv(errors)

