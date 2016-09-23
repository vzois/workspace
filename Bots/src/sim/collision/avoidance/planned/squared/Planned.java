package sim.collision.avoidance.planned.squared;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Scanner;

import sim.structures.Vector;

public class Planned {

	HashMap<Integer,ArrayList<Vector>> map=null;
	String file;
	int agents =0;
	
	public Planned(String file){
		this.file=file;
	}
	
	public HashMap<Integer,ArrayList<Vector>> getMap(){
		return this.map;
	}
	
	public void init(int agents){
		map = new HashMap<Integer,ArrayList<Vector>>();
		for(int i = 0; i < agents; i++){
			map.put(new Integer(i), new ArrayList<Vector>(100000));
		}
	}
	
	public int getAgents(){
		return this.agents;
	}
	
	public void read(){
		Scanner s;
		try {
			s = new Scanner(new File(this.file));
			s.useDelimiter("\n");
			while(s.hasNext())
			{
				String line = s.next();
				if(line.length()>0){
					String []data = line.trim().split(",");
					if(agents==0) this.agents = data.length/2;
					if(this.getMap()==null){ this.init(agents); }
					for(int i=0;i<agents;i++){
						Vector pos = new Vector(Double.valueOf(data[2*i]),Double.valueOf(data[2*i + 1]));
						this.getMap().get(i).add(pos);
					}
				}
			}
			s.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void bulkRead(){
		FileReader fr;
		BufferedReader br;
		try {
			fr = new FileReader(this.file);
			br = new BufferedReader(fr);
			String line;
			String data[];
			
			line = br.readLine();
			data = line.split(",");
			if(agents==0) this.agents = data.length/2;
			if(this.getMap()==null){ this.init(agents); }
			for(int i=0;i<agents;i++){
				Vector pos = new Vector(Double.valueOf(data[2*i]),Double.valueOf(data[2*i + 1]));
				this.getMap().get(i).add(pos);
			}
			
			while ((line = br.readLine()) != null) {
				data = line.split(",");
				for(int i=0;i<agents;i++){
					Vector pos = new Vector(Double.valueOf(data[2*i]),Double.valueOf(data[2*i + 1]));
					this.getMap().get(i).add(pos);
				}
			}
			
			br.close();
			fr.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
}
