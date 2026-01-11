import matplotlib
manager = matplotlib.get_current_fig_manager
manager.window.geometry("500x400+0+0")

figure(1)
plot([1,2,3,4,5])
thismanager = get_current_fig_manager()
print(thismanager)
thismanager.window.SetPosition((500, 0))
show()

