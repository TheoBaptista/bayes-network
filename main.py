from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Este código define uma rede Bayesiana para modelar a relação entre sintomas e uma doença.

# Define a estrutura da rede Bayesiana
# A rede possui 4 variáveis:
model = BayesianNetwork([('Sintoma1', 'Doença'), ('Sintoma2', 'Doença'), ('Sintoma3', 'Doença')])

# A rede possui 3 arrays direcionados, conectando cada nó de sintoma ao nó de doença.
# Isso significa que o nó de doença depende dos sintomas, mas não o contrário.


# Os objetos TabularCPD especificam as distribuições de probabilidade condicional para cada variável. A tabela de
# probabilidade condicional para a variável de doença depende dos valores das variáveis de Sintoma1, Sintoma2 e
# Sintoma3.

cpd_sintoma1 = TabularCPD(variable='Sintoma1', variable_card=2, values=[[0.8], [0.2]])
print(cpd_sintoma1)

cpd_sintoma2 = TabularCPD(variable='Sintoma2', variable_card=2, values=[[0.6], [0.4]])
print(cpd_sintoma2)

cpd_sintoma3 = TabularCPD(variable='Sintoma3', variable_card=2, values=[[0.7], [0.3]])
print(cpd_sintoma3)

cpd_doenca = TabularCPD(variable='Doença', variable_card=2,
                        values=[[0.99, 0.01, 0.05, 0.95, 0.05, 0.95, 0.01, 0.99],
                                [0.01, 0.99, 0.95, 0.05, 0.95, 0.05, 0.99, 0.01]],
                        evidence=['Sintoma1', 'Sintoma2', 'Sintoma3'],
                        evidence_card=[2, 2, 2])
print(cpd_doenca)

# Adiciona as distribuições de probabilidade condicional ao modelo
model.add_cpds(cpd_sintoma1, cpd_sintoma2, cpd_sintoma3, cpd_doenca)

# Verifica se a rede Bayesiana é válida
model.check_model()

# O objeto VariableElimination é criado para realizar inferência na rede
# usando o algoritmo de eliminação de variáveis.
infer = VariableElimination(model)

# Finalmente, o código usa o método de consulta 'infer.query' para calcular a probabilidade da doença
# dado que Sintoma1 e Sintoma3 estão presentes e Sintoma2 está ausente.
# A distribuição de probabilidade resultante sobre a variável Doença é impressa no console.
consulta = infer.query(['Doença'], evidence={'Sintoma1': 1, 'Sintoma2': 1, 'Sintoma3': 1})

# quando executado, o código mostrará a distribuição de probabilidade sobre a variável Doença,
# que mostrará a probabilidade da doença estar presente ou ausente, dadas as informações de sintomas fornecidas.
print(consulta)
