from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork

# Este código define e usa uma rede Bayesiana para modelar a relação entre sintomas e uma doença,
# e usa inferência para calcular a probabilidade da doença dada a presença ou ausência de sintomas específicos.

# Define a estrutura da rede Bayesiana
sintoma_doenca_modelo = BayesianNetwork([('Sintoma1', 'Doença'), ('Sintoma2', 'Doença'), ('Sintoma3', 'Doença')])

# A rede possui 3 arrays direcionados, conectando cada nó de sintoma ao nó de doença.
# Isso significa que o nó de doença depende dos sintomas, mas não o contrário.


# Os objetos TabularCPD especificam as distribuições de probabilidade condicional para cada variável. A tabela de
# probabilidade condicional para a variável de doença depende dos valores das variáveis de Sintoma1, Sintoma2 e
# Sintoma3. Tabelas de Probabilidades Condicional (CPDs)

# Cada CPD é definida como uma tabela que especifica as probabilidades condicionais para a variável correspondente.
# No caso desses sintomas, como são variáveis binárias,
# cada CPD contém apenas uma coluna com a probabilidade de cada valor possível.
cpd_sintoma1 = TabularCPD(variable='Sintoma1', variable_card=2, values=[[0.8], [0.2]])
cpd_sintoma2 = TabularCPD(variable='Sintoma2', variable_card=2, values=[[0.6], [0.4]])
cpd_sintoma3 = TabularCPD(variable='Sintoma3', variable_card=2, values=[[0.7], [0.3]])
cpd_doenca = TabularCPD(variable='Doença', variable_card=2,
                        values=[[0.99, 0.01, 0.05, 0.95, 0.05, 0.95, 0.01, 0.99],
                                [0.01, 0.99, 0.95, 0.05, 0.95, 0.05, 0.99, 0.01]],
                        evidence=['Sintoma1', 'Sintoma2', 'Sintoma3'],
                        evidence_card=[2, 2, 2])

# A última linha define a CPD da doença, que depende dos três sintomas anteriores.
# Essa CPD é definida como uma tabela com 8 valores,
# que correspondem às combinações possíveis dos valores dos sintomas 1, 2 e 3.
# As duas linhas da tabela correspondem aos dois possíveis valores da variável 'Doença',
# e a ordem dos valores nas linhas é determinada pela ordem em que os valores dos sintomas aparecem na lista 'evidence'.
# A lista 'evidence_card' especifica o número de valores possíveis para cada um dos sintomas (no caso, 2 para cada).

# Adiciona as distribuições de probabilidade condicional ao modelo
sintoma_doenca_modelo.add_cpds(cpd_sintoma1, cpd_sintoma2, cpd_sintoma3, cpd_doenca)

# Verifica se a rede Bayesiana é válida
sintoma_doenca_modelo.check_model()

# O objeto VariableElimination é criado para realizar inferência na rede
infer = VariableElimination(sintoma_doenca_modelo)

# Finalmente, o código usa o método de consulta 'infer.query' para calcular a probabilidade da doença
# dado que Sintoma1 e Sintoma3 estão presentes e Sintoma2 está ausente.
consulta = infer.query(['Doença'], evidence={'Sintoma1': 1, 'Sintoma2': 0, 'Sintoma3': 1})

# Imprime a distribuição de probabilidade resultante sobre a variável Doença
print(consulta)

# a probabilidade de o paciente ter a doença é de 0.0500,
# enquanto a probabilidade de não ter a doença é de 0.9500.
