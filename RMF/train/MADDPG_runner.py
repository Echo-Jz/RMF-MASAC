
from .run import TrainerBase
from RMF.agents.MAgentMADDPG import AgentMADDPG



class MADDPG_Trainer(TrainerBase):
    def __init__(self,arglist,envclass):
        super().__init__(AgentMADDPG,envclass,arglist)



