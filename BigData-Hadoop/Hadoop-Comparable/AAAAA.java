// package com.hadoop_comparable;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class AAAAA
{
  public void addelements(ArrayList<student_info> studname_list)
  {
    studname_list.add(new student_info("sam"));
    studname_list.add(new student_info("vam"));
    studname_list.add(new student_info("mam"));
    studname_list.add(new student_info("jam"));
    studname_list.add(new student_info("pam"));
    studname_list.add(new student_info("lam"));
    studname_list.add(new student_info("wam"));
    studname_list.add(new student_info("vam"));
  }
  
  public void addelements_1(ArrayList<String> studname_list_1)
  {
    studname_list_1.add("sam");
    studname_list_1.add("vam");
    studname_list_1.add("mam");
    studname_list_1.add("jam");
    studname_list_1.add("pam");
    studname_list_1.add("lam");
    studname_list_1.add("wam");
    studname_list_1.add("vam");
  }
  
  public void dispelements(ArrayList<student_info> studname_list)
  {
    for (student_info eachstud : studname_list) {
      System.out.println(eachstud);
    }
  }
  
  public void dispelements_1(ArrayList<String> studname_list_1)
  {
    for (String eachstud : studname_list_1) {
      System.out.println(eachstud);
    }
  }
  
  public static void main(String[] args)
  {
    List<student_info> studname_list = new ArrayList();
    List<String> studname_list_1 = new ArrayList();
    AAAAA stud_list = new AAAAA();
    
    stud_list.addelements((ArrayList)studname_list);
    Collections.sort(studname_list);
    
    stud_list.dispelements((ArrayList)studname_list);
    
    System.out.println("\n");
    
    stud_list.addelements_1((ArrayList)studname_list_1);
    Collections.sort(studname_list_1);
    
    stud_list.dispelements_1((ArrayList)studname_list_1);
  }
}
