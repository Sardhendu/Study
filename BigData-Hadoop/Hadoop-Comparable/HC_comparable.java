package com.hadoop_comparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import org.apache.hadoop.io.Writable;

public class HC_comparable
  implements Writable
{
  private int numElements;
  private double average;
  List<HC_comparable> list = new ArrayList();
  
  public void set(int numElements, double average)
  {
    System.out.println("I am inside the set method of HC_comparable");
    this.numElements = numElements;
    this.average = average;
  }
  
  public int getNumElements()
  {
    System.out.println("I am inside the getNumElements method of HC_comparable");
    return this.numElements;
  }
  
  public double getAverage()
  {
    System.out.println("I am inside the getAverage method of HC_comparable");
    return this.average;
  }
  
  public void readFields(DataInput input)
    throws IOException
  {
    System.out.println("I am inside the readFields method of HC_comparable");
    this.numElements = input.readInt();
    this.average = input.readDouble();
  }
  
  public void write(DataOutput output)
    throws IOException
  {
    System.out.println("I am inside the write method of HC_comparable");
    output.writeInt(this.numElements);
    output.writeDouble(this.average);
  }
  
  public String toString()
  {
    return this.average + " , " + this.numElements;
  }
}
