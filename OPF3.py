import pandas as pd
from pyomo.environ import *
from pyomo.environ import SolverFactory
import network

# Função para criar o modelo Pyomo usando os coeficientes de sensibilidade
def create_model(network):
    model = ConcreteModel()
    model.name = network.name

    # Sets
    model.periods = range(network.num_instants)
    model.nodes = range(len(network.nodes))
    model.generators = range(len(network.generators))
    model.branches = range(len(network.branches))

    # Constraints
    model.vmin = Param(initialize=0.95)  # Limite mínimo de tensão
    model.vmax = Param(initialize=1.05)  # Limite máximo de tensão
    model.sensitivity_p = Param(model.nodes, model.nodes, initialize=lambda model, i, j: network.sv_p[str(i)][str(j)])
    model.sensitivity_q = Param(model.nodes, model.nodes, initialize=lambda model, i, j: network.sv_q[str(i)][str(j)])


    # Decision Variables
    model.s_up = Var(model.N, model.T, domain=NonNegativeReals)  # Variável de folga para desvio de tensão para cima
    model.s_down = Var(model.N, model.T, domain=NonNegativeReals)  # Variável de folga para desvio de tensão para baixo
    model.delta_p = Var(model.N, model.T, domain=Reals)  # Mudanças de potência ativa em cada barra
    model.delta_q = Var(model.N, model.T, domain=Reals)  # Mudanças de potência reativa em cada barra
    model.V_actual = Var(model.nodes, model.periods, domain=NonNegativeReals)  # Define actual voltage variable


    # Restrição de magnitude de tensão incorporando coeficientes de sensibilidade
    def voltage_magnitude_constraint(model, n, t):
        v_base = 1.0 + sum(
            model.sensitivity_p[n, m] * model.delta_p[m, t] + model.sensitivity_q[n, m] * model.delta_q[m, t]
            for m in model.N
        )
        return model.V_actual[n, t] == v_base + model.s_up[n, t] - model.s_down[n, t]

    model.voltage_magnitude_constraint = Constraint(model.N, model.T, rule=voltage_magnitude_constraint)

    # Limites de tensão em cada nó para cada período de tempo
    def voltage_limit_rule_min(model, n, t):
        return model.vmin <= model.V_actual[n, t]

    def voltage_limit_rule_max(model, n, t):
        return model.V_actual[n, t] <= model.vmax

    model.voltage_limit_constraint_min = Constraint(model.N, model.T, rule=voltage_limit_rule_min)
    model.voltage_limit_constraint_max = Constraint(model.N, model.T, rule=voltage_limit_rule_max)

    # Função objetivo: Minimizar a soma das variáveis de folga
    def objective_rule(model):
        return sum(model.s_up[n, t] + model.s_down[n, t] for n in model.N for t in model.T)
    
    model.obj = Objective(rule=objective_rule, sense=minimize)

    return model

# Função para resolver o modelo
def solve_model(model, tee=True):
    solver = SolverFactory('cplex')  
    result = solver.solve(model, tee=tee)  
    return result


