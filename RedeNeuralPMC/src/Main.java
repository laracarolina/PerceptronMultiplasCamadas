import java.util.*;
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
        int op = 0;
        double x1, x2, x3, x4;
        Scanner s = new Scanner(System.in);
        MultilayerPerceptron mlp = new MultilayerPerceptron();
        System.out.println("\n>> Perceptron Multiplas Camadas como auxiliador no processamento de bebidas <<");
        do{
         System.out.println("\n1 - Entrar com uma amostra");
         System.out.println("2 - Utilizar amostras do arquivo");
         System.out.println("3 - Sair");
         op = s.nextInt();
         if(op == 1){
            System.out.println("\nTeor de agua: ");
            x1 = s.nextDouble();
            System.out.println("\nGrau de acidez: ");
            x2 = s.nextDouble();
            System.out.println("\nTemperatura: ");
            x3 = s.nextDouble();
            System.out.println("\nTensão interfacial: ");
            x4 = s.nextDouble();
            x = mlp.treinamentoBackpropagation2(0.9, x1, x2, x3, x4);
         }
         else if(op == 2){
        System.out.println("\n------------- Treinamento sem momento: -----------------\n\n");
        x = mlp.treinamentoBackpropagation(0.0);
        System.out.println("\n------------- Treinamento com momento: -----------------\n\n");
        x = mlp.treinamentoBackpropagation(0.9);
         }
        } while(op!=3);
    }
}
