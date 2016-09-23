package sim.exec;

import sim.graphics.World;

public class WorldThread extends Thread {
	boolean run=true;
	World world;
	
	public WorldThread(World world){
		this.world = world;
	}
	
	public void run(){
		while(run){
			try {
				Thread.sleep(0);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void stopT(){
		run = false;
	}
}
