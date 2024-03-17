import logger

from bbrl.agents.agent import Agent

class LoggerAgent(Agent):
    def __init__(self, work_dir, cfg):
        super().__init__()
        self.logger = logger.Logger(work_dir, cfg)
        # Ajout pour suivre les métriques au fil du temps
        self.metrics_history = {}
    
    def forward(self, workspace, t, **kwargs):
        # Récupération et enregistrement des métriques
        if 'metrics' in kwargs and 'category' in kwargs:
            metrics = kwargs['metrics']
            category = kwargs['category']
            self.logger.log(metrics, category)
            # Stockage des métriques pour une utilisation future
            for key, value in metrics.items():
                if key not in self.metrics_history:
                    self.metrics_history[key] = []
                self.metrics_history[key].append(value)
            # Mise à jour de workspace avec des informations potentiellement utiles pour d'autres agents
            workspace.set(f"{category}_metrics", t, metrics)
            
    def get_metrics_history(self, metric_name):
        return self.metrics_history.get(metric_name, [])

    def isVideoEnable(self):
        return self.logger.video is not None

    