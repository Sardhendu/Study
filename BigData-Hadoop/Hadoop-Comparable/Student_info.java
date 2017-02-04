package com.hadoop_comparable;

class student_info
  implements Comparable<student_info>
{
  private String stud_name;
  
  public student_info(String stud_name)
  {
    this.stud_name = stud_name.toString();
  }
  
  public String toString()
  {
    return this.stud_name;
  }
  
  public int compareTo(student_info stud_name_1)
  {
    return this.stud_name.compareTo(stud_name_1.stud_name);
  }
}
