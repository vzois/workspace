package sim.exec;

import java.util.ArrayList;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

import sim.elements.Bot;
import sim.graphics.World;

public class BalancedThread extends Thread {
	World world;
	CyclicBarrier cb;
	ArrayList<Bot> bots;
	boolean run;
	int id;
	
	public BalancedThread(World world,CyclicBarrier cb){
		this.world = world;
		this.cb = cb;
		this.run = true;
		this.bots = new ArrayList<Bot>();
	}
	
	public void addBot(Bot bot){
		this.bots.add(bot);
	}
	
	public void setID(int id){
		this.id = id;
	}
	
	public void run(){
		while(run){
			try {
				Thread.sleep(world.getSpeed());
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
			if(id==0){ 
				world.repaint();
				world.time++;
				world.getTextField().setText(world.time+"");
				world.initKDTree();
			}
			
			for(Bot bot : bots) world.insertObject(bot);
			synchronize();
			
			for(Bot bot : bots){
				bot.setNeighbors(world.getObjects(bot));
				bot.act();
				bot.move(world.getLX(),world.getLY());
			}
			synchronize();
		}
	}
	
	public void synchronize(){
		try {
			cb.await();
		} catch (InterruptedException | BrokenBarrierException e) {
			e.printStackTrace();
		}
	}
	
	public void stopT(){
		this.run = false;
	}
}
