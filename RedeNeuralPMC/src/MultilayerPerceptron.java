/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Lara
 */
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

public class MultilayerPerceptron {

    // a rede possui 4 entradas, uma camada oculta com 15 neuronios e a camada de saída com 3 neuronios
    public static final int numeroEntradas = 4;
    private final int numeroNeuroniosOcultos = 15;
    public static final int numeroNeuroniosSaida = 3;
    private final double taxaDeAprendizagem = 0.1;
    private final double precisao = 0.000001;
    private final double parametroMomento = 0.9;
    private final double BETA = 1.0;

    private int numEpocas;
    private double momento;
    private double[] entradas;
    private double[][] pesosCamadaOcultaInicial; // contém os pesos da camada oculta iniciados aleatoriamente
    private double[][] pesosCamadaOculta; // contém os pesos atuais da camada oculta no processamento da amostra n
    private double[][] pesosCamadaOcultaProximo; // contém os pesos da camada oculta após ajustes, no processamento da amostra n
    private double[][] pesosCamadaOcultaAnterior; // contém os pesos da camada oculta ajustados no processamento da amostra n-1
    private double[][] pesosCamadaSaidaInicial; // contém os pesos da camada de saída iniciados aleatoriamente
    private double[][] pesosCamadaSaida; // contém os pesos atuais da camada de saida no processamento da amostra n
    private double[][] pesosCamadaSaidaProximo; // contém os pesos da camada de saida após ajustes, no processamento da amostra n
    private double[][] pesosCamadaSaidaAnterior;  // contém os pesos da camada de saida ajustados no processamento da amostra n-1
    private double[] potencialCamadaOculta; // contem os potenciais de ativação dos neuronios da camada oculta
    private double[] saidaCamadaOculta; // contem as saidas dos neuronios da camada oculta
    private double[] potencialCamadaSaida; // contem os potenciais de ativação dos neuronios da camada de saida
    private double[] saidaCamadaSaida; // contem as saidas dos neuronios da camada de saida
    private double[] saidaEsperada; // contem as saidas esperadas para a rede
    private double[] gradienteCamadaSaida; // contem os gradientes dos neuronios da camada de saida
    private double[] gradienteCamadaOculta; // contem os gradientes dos neuronios da camada oculta

    // **** Cada linha das matrizes de pesos contém os pesos relativos a um neuronio. Por exemplo, pesosCamadaOculta é uma matriz
    // com 15 linhas pois há 15 neuronios e 4 colunas pois a rede tem 4 entradas
    public MultilayerPerceptron() {
        Random random;

        entradas = new double[numeroEntradas + 1]; // +1 pois é utilizado um vies com valor igual a -1
        pesosCamadaOcultaInicial = new double[numeroNeuroniosOcultos][numeroEntradas + 1];
        pesosCamadaOculta = new double[numeroNeuroniosOcultos][numeroEntradas + 1];
        pesosCamadaOcultaAnterior = new double[numeroNeuroniosOcultos][numeroEntradas + 1];
        pesosCamadaOcultaProximo = new double[numeroNeuroniosOcultos][numeroEntradas + 1];
        pesosCamadaSaidaInicial = new double[numeroNeuroniosSaida][numeroNeuroniosOcultos + 1];
        pesosCamadaSaida = new double[numeroNeuroniosSaida][numeroNeuroniosOcultos + 1];
        pesosCamadaSaidaAnterior = new double[numeroNeuroniosSaida][numeroNeuroniosOcultos + 1];
        pesosCamadaSaidaProximo = new double[numeroNeuroniosSaida][numeroNeuroniosOcultos + 1];
        potencialCamadaOculta = new double[numeroNeuroniosOcultos + 1];
        saidaCamadaOculta = new double[numeroNeuroniosOcultos + 1];
        potencialCamadaSaida = new double[numeroNeuroniosSaida];
        saidaCamadaSaida = new double[numeroNeuroniosSaida];
        saidaEsperada = new double[numeroNeuroniosSaida];
        gradienteCamadaSaida = new double[numeroNeuroniosSaida];
        gradienteCamadaOculta = new double[numeroNeuroniosOcultos];

        //Iniciando pesos sinapticos aleatoriamente com valores entre 0 e 1
        random = new Random();

        for (int i = 0; i < numeroNeuroniosOcultos; i++) {
            for (int j = 0; j < numeroEntradas + 1; j++) {
                pesosCamadaOcultaInicial[i][j] = random.nextDouble();
            }
        }

        for (int i = 0; i < numeroNeuroniosSaida; i++) {
            for (int j = 0; j < numeroNeuroniosOcultos + 1; j++) {
                pesosCamadaSaidaInicial[i][j] = random.nextDouble();
            }
        }

        transferirConteudo(pesosCamadaOcultaInicial, pesosCamadaOculta);
        transferirConteudo(pesosCamadaSaidaInicial, pesosCamadaSaida);

    }

    public boolean treinamentoBackpropagation(boolean fatorMomento) {
        FileReader arq;
        BufferedReader lerArq;
        String linha;
        double erroAtual;
        double erroAnterior;
        long tempInicial;

        if (!fatorMomento) { // caso momento seja 'false' o parametro terá valor 0, ou seja, a rede será treinada sem inclusao deste parametro
            momento = 0D;

        } else { // caso momento seja 'true' o parametro terá valor 0.9
            momento = parametroMomento;

        }

        transferirConteudo(pesosCamadaOcultaInicial, pesosCamadaOculta);
        transferirConteudo(pesosCamadaSaidaInicial, pesosCamadaSaida);
        // para trabalhar com o parametro momento é necessário guardar os pesos ajustados no processamento da amostra anterior
        transferirConteudo(pesosCamadaOcultaInicial, pesosCamadaOcultaAnterior);
        transferirConteudo(pesosCamadaSaidaInicial, pesosCamadaSaidaAnterior);

        tempInicial = System.currentTimeMillis();
        numEpocas = 0;
        erroAtual = erroQuadraticoMedio(); // calcula o erro da rede com o vetor de pesos inicial

        try {
            do {

                this.numEpocas++;
                erroAnterior = erroAtual;

                arq = new FileReader("C:\\Users\\Lara\\Documents\\NetBeansProjects\\RedeNeural\\treinamento.txt");
                lerArq = new BufferedReader(arq);

                //uma linha do arquivo contem 4 valores de entrada para a rede e as saidas esperadas para os 3 neuronios da camada de saida, separados por espaço. 
                linha = lerArq.readLine(); // le a primeira linha ( contem apenas x1 x2 x3 x4 d1 d4 d3)
                linha = lerArq.readLine(); // le a proxima linha com os primeiros valores para o treinamento e as saidas desejadas
                // este é o laço que corresponde ao processamento de todas as amostras de treinamento
                while (linha != null) { // enquanto o arquivo não chegar ao fim, ou seja, enquanto existirem amostras

                    obterDadosSeparados(linha); // obtem os 4 valores de entrada separadamente e os armazena no vetor entradas e os 3
                    // valores de saida dos neuronios de saida e os armazena no vetor saidaEsperada

                    calcularSaidas(); // calcula as saidas da rede para a amostra em processamento

                    ajustarPesos(); // ajusta os pesos da rede

                    linha = lerArq.readLine(); // le a proxima linha do arquivo que contem a proxima amostra a ser processada
                }

                arq.close();
                erroAtual = erroQuadraticoMedio(); // calcula o erro da epoca
                //System.out.println("\n "+numEpocas+"  "+erroAtual);

            } while ((Math.abs(erroAtual - erroAnterior) > precisao) && (numEpocas < 1000));

            System.out.println("\nTempo = " + (System.currentTimeMillis() - tempInicial) / 1000D);
            System.out.println("\nErro atual: " + erroAtual);
            System.out.println("\nErro anterior: " + erroAnterior);
            System.out.println("\n\nO criterio de parada foi atingido com numero de epocas igual a " + numEpocas);

        } catch (FileNotFoundException ex) {
            return false;
        } catch (IOException ex) {
            return false;
        }
        //imprimirPesos();
        teste();
        return true;
    }

    // calcula a porcentagem de acertos da rede após o fim do seu treinamento.
    public void teste() {
        FileReader arq;
        BufferedReader lerArq;
        String linha;
        boolean erro;
        int erros, numAmostras;
        double porcentagemAcertos;
        numAmostras = 0;
        erros = 0;

        try {
            arq = new FileReader("C:\\Users\\Lara\\Documents\\NetBeansProjects\\RedeNeural\\treinamento.txt");
            lerArq = new BufferedReader(arq);

            linha = lerArq.readLine();
            if (linha.contains("x1")) {
                linha = lerArq.readLine();
            }

            // para cada amostra, a saida desejada de cada neuronio de saida é comparada a saida obtida de cada neuronio de saida
            // caso pelo menos uma saida seja diferente da esperada, o contador de erros é incrementado
            while (linha != null) {
                numAmostras++;
                obterDadosSeparados(linha);
                calcularSaidas();
                erro = false;
                //System.out.println("\n\nAmostra "+numAmostras+":");
                for (int i = 0; i < numeroNeuroniosSaida; i++) {

                    //System.out.println("d"+i+": "+saidaEsperada[i]+" y"+i+": "+posProcessamento(saidaCamadaSaida[i]));
                    if (saidaEsperada[i] != posProcessamento(saidaCamadaSaida[i])) {
                        erro = true;
                    }
                }

                if (erro == true) {
                    erros++;
                }

                linha = lerArq.readLine();
            }

            arq.close();
            porcentagemAcertos = (((numAmostras - erros) * 100D) / numAmostras); // calculo da porcentagem de acertos

            System.out.println("\n\n A rede obteve uma porcentagem de acertos igual a " + porcentagemAcertos + "\n");
            System.out.println(" Amostras = " + numAmostras);
            System.out.println("acertos = " + (numAmostras - erros));
        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        }
    }

    // calcula o erro Quadratico Médio de todas as amostras considerando o vetor de pesos ajustado
    private double erroQuadraticoMedio() {
        FileReader arq;
        BufferedReader lerArq;
        int numAmostras = 0;
        double eqm, erroAmostra;
        String linha;
        eqm = 0D;

        try {
            arq = new FileReader("C:\\Users\\Lara\\Documents\\NetBeansProjects\\RedeNeural\\treinamento.txt");
            lerArq = new BufferedReader(arq);

            linha = lerArq.readLine();

            linha = lerArq.readLine();

            // para cada amostra calcula a saida desejada - saida obtida e eleva ao quadrado
            while (linha != null) {
                numAmostras++;
                obterDadosSeparados(linha); // separa os dados da amostra e os valores de saida desejados

                calcularSaidas(); // calcula as saidas da rede para a amostra

                erroAmostra = 0D;
                for (int i = 0; i < saidaCamadaSaida.length; i++) {
                    // o erro de cada amostra é o somatorio da saida esperada para cada neuronio de saida menos a saida obtida para cada
                    // neuronio de saida elevado ao quadrado
                    erroAmostra = erroAmostra + Math.pow((double) (saidaEsperada[i] - saidaCamadaSaida[i]), 2D);
                }
                eqm = eqm + (erroAmostra / 2D);

                linha = lerArq.readLine();
            }

            arq.close();
            eqm = eqm / (double) numAmostras;

        } catch (FileNotFoundException ex) {
        } catch (IOException ex) {
        }

        return eqm;
    }

    private void obterDadosSeparados(String linha) {
        String[] vetor; // cada posicao deste vetor contem um valor contido em uma linha do arquivo
        // este vetor contem os valores de x1, x2, x3, x4, que são as entradas, e os valores de d1, d2, d3 que são as saidas
        // desejadas para os neuronios de saida
        int i;

        vetor = linha.split("\\s+"); // split faz com que os valores na linha separados por espaço sejam colocados, cada um, em uma posição do vetor
        i = 0;

        if (vetor[0].equals("")) { // caso o valor da primeira posição seja um espaco em branco, i começa em 1 para desconsiderar tal espaço
            i = 1;
        }

        entradas[0] = -1.0; // este é o vies
        // preenchendo o vetor de entradas
        for (int j = 1; j <= numeroEntradas; j++) {
            entradas[j] = Double.parseDouble(vetor[i++].replace(",", ".")); // converte a string da posiçao do vetor para um double e coloca tal valor em entradas
        }
        // preenchendo o vetor de saidasEsperadas
        for (int j = 0; j < numeroNeuroniosSaida; j++) {
            saidaEsperada[j] = Double.parseDouble(vetor[i++].replace(",", "."));
        }
    }

    // aqui são calculados os potenciais de ativação dos neuronios ocultos e de saida e suas saidas
    private void calcularSaidas() {
        double valorParcial;

        // Calculando potenciais de ativação e saidas da camada oculta
        saidaCamadaOculta[0] = -1D;
        potencialCamadaOculta[0] = -1D;

        for (int i = 1; i < saidaCamadaOculta.length; i++) {
            valorParcial = 0D;

            // calculo do potencial de ativação do neuronio i
            for (int j = 0; j < entradas.length; j++) {
                valorParcial += entradas[j] * pesosCamadaOculta[i - 1][j]; // cada sinal é multiplicado pelo seu respectivo peso.
            }

            potencialCamadaOculta[i] = valorParcial;
            saidaCamadaOculta[i] = funcaoLogistica(valorParcial); // a saida é a função de ativação aplicada ao potencial de ativação
        }

        //Calculando potenciais de ativação e  saidas da camada de saída
        for (int i = 0; i < saidaCamadaSaida.length; i++) {
            valorParcial = 0D;

            // calculo do potencial de ativação do neuronio i 
            for (int j = 0; j < saidaCamadaOculta.length; j++) {
                valorParcial += saidaCamadaOculta[j] * pesosCamadaSaida[i][j]; // cada sinal é multiplicado pelo seu respectivo peso
            }

            potencialCamadaSaida[i] = valorParcial;
            saidaCamadaSaida[i] = funcaoLogistica(valorParcial);
        }
    }

    private void ajustarPesos() {

        //Ajustando pesos sinapticos da camada de saida
        // o primeiro passo é calcular os gradientes dos neuronios de saida
        for (int i = 0; i < gradienteCamadaSaida.length; i++) {
            // o gradiente de um neuronio de saida é igual ao erro deste neuronio multiplicado pela função de ativação
            // aplicada ao potencial de ativação deste neuronio
            gradienteCamadaSaida[i] = (saidaEsperada[i] - saidaCamadaSaida[i]) * funcaoLogisticaDerivada(potencialCamadaSaida[i]);

            // ajustando os pesos dos neuronios de saida
            // A formula de ajuste de um peso, com a inclusão do momento, corresponde ao peso atual somado ao delta.
            // O delta é a taxa de aprendizagem multiplicada pel gradiente do neuronio e pelo sinal que circula no link,
            // que no caso é a saida da camada oculta, somado ao fator momento o qual é multiplicado pelo valor do peso no processamento
            // da amostra anterior
            for (int j = 0; j < numeroNeuroniosOcultos + 1; j++) {
                pesosCamadaSaidaProximo[i][j] = pesosCamadaSaida[i][j]
                        + (momento * (pesosCamadaSaida[i][j] - pesosCamadaSaidaAnterior[i][j]))
                        + (taxaDeAprendizagem * gradienteCamadaSaida[i] * saidaCamadaOculta[j]);
            }
        }

        //Ajustando pesos sinapticos da camada escondida
        // primeiramente são calculados os gradientes dos neuronios
        for (int i = 0; i < gradienteCamadaOculta.length; i++) {
            gradienteCamadaOculta[i] = 0D;
            for (int j = 0; j < numeroNeuroniosSaida; j++) {
                // Para calcular o gradiente do neuronio i, primeiramente é o calculado o valor do somatorio abaixo,
                // que corresponde a cada gradiente dos neuronios da proxima camada multiplicado pelo peso do link que conecta
                // o neuronio i ao neuronio da prox camada.
                gradienteCamadaOculta[i] += gradienteCamadaSaida[j] * pesosCamadaSaida[j][i + 1];
            }
            // o somatorio é multiplicado pela função de ativação aplicada ao potencial de ativação do neuronio
            gradienteCamadaOculta[i] *= funcaoLogisticaDerivada(potencialCamadaOculta[i + 1]);

            // a formula de ajuste dos pesos é igual a formula utilizada para ajustar os pesos do neuronio de saida,
            // a diferença esta apenas no calculo dos gradientes que é feito levando em consideração os gradientes dos neuronios
            // da camada subsequente
            for (int j = 0; j < numeroEntradas + 1; j++) {
                pesosCamadaOcultaProximo[i][j] = pesosCamadaOculta[i][j]
                        + (momento * (pesosCamadaOculta[i][j] - pesosCamadaOcultaAnterior[i][j]))
                        + (taxaDeAprendizagem * gradienteCamadaOculta[i] * entradas[j]);
            }
        }

        // os pesos atuais passam a ser os pesos ajustados e os pesos anteriores passam a ser os pesos anteriores ao ajuste
        transferirConteudo(pesosCamadaOculta, pesosCamadaOcultaAnterior);
        transferirConteudo(pesosCamadaOcultaProximo, pesosCamadaOculta);
        transferirConteudo(pesosCamadaSaida, pesosCamadaSaidaAnterior);
        transferirConteudo(pesosCamadaSaidaProximo, pesosCamadaSaida);
    }

    //transfere o conteudo da matriz origem para a matriz destino
    private void transferirConteudo(double[][] origem, double[][] destino) {
        for (int i = 0; i < origem.length; i++) {
            for (int j = 0; j < origem[i].length; j++) {
                if (destino.length > i) {
                    if (destino[i].length > j) {
                        destino[i][j] = origem[i][j];
                    }
                }
            }
        }
    }

    // função de ativação
    private double funcaoLogistica(double valor) {
        return 1D / (1D + Math.pow(Math.E, -1D * BETA * valor));
    }

    // derivada da função de ativação
    private double funcaoLogisticaDerivada(double valor) {
        return (BETA * Math.pow(Math.E, -1D * BETA * valor)) / Math.pow((Math.pow(Math.E, -1D * BETA * valor) + 1D), 2D);
    }

    private int posProcessamento(double valor) {
        int resposta = 0;

        if (valor >= 0.5) {
            resposta = 1;
        }

        return resposta;
    }

    private void imprimirPesos() {
        int i, j;

        System.out.println("\n ------- Pesos da Camada Oculta ---------");
        for (i = 0; i < numeroNeuroniosOcultos; i++) {
            System.out.println("\n\nPesos do neuronio " + (i + 1) + ":");
            for (j = 0; j < numeroEntradas + 1; j++) {
                System.out.print(" " + pesosCamadaOculta[i][j]);
            }
        }

        System.out.println("\n\n ------- Pesos da Camada de Saida ---------");
        for (i = 0; i < numeroNeuroniosSaida; i++) {
            System.out.println("\n\nPesos do neuronio " + (i + 1) + ":");
            for (j = 0; j < numeroNeuroniosOcultos + 1; j++) {
                System.out.print(" " + pesosCamadaSaida[i][j]);
            }
        }
    }

}
