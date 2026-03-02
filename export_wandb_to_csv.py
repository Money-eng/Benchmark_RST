import wandb
api = wandb.Api()


run = api.run("lgand-universit-de-montpellier/chronoRoot_logs")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
metrics_dataframe.to_csv("metrics.csv")