/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author Lara
 */
public class Main {

    public static void main(String args[]) {
        boolean x;
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        System.out.println("\n------------- Treinamento sem momento: -----------------\n\n");
        x = mlp.treinamentoBackpropagation(false);
        System.out.println("\n------------- Treinamento com momento: -----------------\n\n");
        x = mlp.treinamentoBackpropagation(true);
    }
}
