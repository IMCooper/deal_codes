function [] = plot_deal_ii_results()
clear
close all


% UNIFORM:

uniform_results{1}=[        0     1    24 5.94525548e-02 ;
    1     8   108 4.56114313e-02 ;
    2    64   600 3.46584775e-02 ;
    3   512  3888 1.87987348e-02 ;
    4  4096 27744 8.26223509e-03 ];

uniform_results{2}=[    0     1    108 7.12426627e-02 ;
    1     8    600 3.03413056e-02 ;
    2    64   3888 1.48237690e-02 ;
    3   512  27744 4.16693836e-03 ;
    4  4096 209088 7.42319323e-04  ];
    
uniform_results{3} = [    0     1   288 7.79680925e-02 ;
    1     8  1764 1.53672547e-02 ;
    2    64 12168 4.66984610e-03 ;
    3   512 90000 6.74526211e-04 ];

uniform_results{4} = [    0     1    600 8.13784294e-02 
    1     8   3888 6.98494705e-03 
    2    64  27744 1.32667854e-03 
    3   512 209088 1.33948732e-04  ];

graded_results{1} = [    0     1    24 5.94525548e-02 ;
    1     8   108 4.56114313e-02 ;
    2    57   576 3.46627978e-02 ;
    3   246  2256 1.89187533e-02 ;
    4   792  6648 8.74550190e-03 ;
    5  2374 18492 4.94297546e-03 ];

graded_results{2}=[    0     1    108 7.12426627e-02 
    1     8    600 3.03413056e-02 
    2    57   3612 1.48239211e-02 
    3   246  14784 4.16951548e-03 
    4   792  45336 7.61382959e-04 
    5  2374 130488 2.19250674e-04];

graded_results{3}=[    0     1    288 7.79680925e-02 ;
    1     8   1764 1.53672547e-02 ;
    2    57  11160 4.66984827e-03 ;
    3   246  46440 6.74625779e-04 ;
    4   792 144576 7.13969008e-05];

graded_results{4}=[    0     1    600 8.13784294e-02 ;
    1     8   3888 6.98494705e-03 ;
    2    57  25272 1.32667892e-03 ;
    3   246 106080 1.33950734e-04    ];

% plot :
colour_list={'.-b' '.-g' '.-r' '.-k' ...
                '.--b' '.--g' '.--r' '.--k'  };
p_legend={'uniform p=0','uniform p=1','uniform p=2','uniform p=3' ...
          'graded p=0','graded p=1','graded p=2','graded p=3' };
h1=figure;
hold on;
% uniform:
for i = 1:4
    plot_data_x=uniform_results{i}(:,3);
    plot_data_y=uniform_results{i}(:,4);
    plot(plot_data_x,plot_data_y,colour_list{i},'LineWidth',2,'MarkerSize',10)
end
% graded:
for i = 1:4
    plot_data_x=graded_results{i}(:,3);
    plot_data_y=graded_results{i}(:,4);
    plot(plot_data_x,plot_data_y,colour_list{i+4},'LineWidth',2,'MarkerSize',10)
end

set(gca,'XScale','log');
set(gca,'YScale','log');
xlabel('DoFs');
ylabel('L2-norm of Error');
legend(p_legend);
title_str=strcat('Uniform vs Graded: h-refinement');
title(title_str);

filename=strcat('uniform_vs_graded','_h-ref');
saveas(h1,filename,'fig');
saveas(h1,filename,'epsc2');


% Fixed grade, increase p:
h2=figure;
hold on
h_legend={'uniform level 1','uniform level 2','uniform level 3','uniform level 4' ...
          'graded level 1','graded level 2','graded level 3','graded level 4' };
for i=1:4
    for j=1:4
        plot_data_x(j)=uniform_results{j}(i,3);
        plot_data_y(j)=uniform_results{j}(i,4);
    end
    plot(plot_data_x,plot_data_y,colour_list{i},'LineWidth',2,'MarkerSize',10)
end
for i=1:4
    for j=1:4
        plot_data_x(j)=graded_results{j}(i,3);
        plot_data_y(j)=graded_results{j}(i,4);
    end
    plot(plot_data_x,plot_data_y,colour_list{i+4},'LineWidth',2,'MarkerSize',10)
end
set(gca,'XScale','log');
set(gca,'YScale','log');
xlabel('DoFs');
ylabel('L2-norm of Error');
legend(h_legend);
title_str=strcat('Uniform vs Graded: p-refinement');
title(title_str);

filename=strcat('uniform_vs_graded','_p-ref');
saveas(h2,filename,'fig');
saveas(h2,filename,'epsc2');

% filename=strcat(save_in,'_h-ref.fig');
% saveas(h1,filename,'fig');
% close(h1);

% function []=do_the_plot(data_in,title_in,save_in)
% 
% colour_list={'.-b' '.-g' '.-r' '.-k' '.-y' '.-m' ...
%                 '.--b' '.--g' '.--r' '.--k' '.--y' '.--m'    };
% p_legend={'p=0','p=1','p=2','p=3','p=4','p=5'};
% h_legend={'level 1','level 2','level 3','level 4','level 5','level 6'};
% 
% max_p=length(data_in)
% % plot fixed p refine h:
% h1=figure;
% hold on
% % ax=axes;
% for i=1:max_p
%     plot_data_x=data_in{i}(:,3);
%     plot_data_y=data_in{i}(:,4);
%     plot(plot_data_x,plot_data_y,colour_list{i},'MarkerSize',10)
% end
% set(gca,'XScale','log');
% set(gca,'YScale','log');
% legend({p_legend{1:max_p}});
% title_str=strcat(title_in,' h-refinement');
% title(title_str);
% 
% filename=strcat(save_in,'_h-ref.fig');
% saveas(h1,filename,'fig');
% close(h1);

% plot fixed h, refine p.
% h2=figure;
% hold on
% for j=1:6
% for i=1:6
%     datax(i,j)=data_in(j,3,i);
%     datay(i,j)=data_in(j,4,i);
% end
% end
% plot(datax,datay,'.-','MarkerSize',10)
% set(gca,'XScale','log');
% set(gca,'YScale','log');
% legend(h_legend);
% title_str=strcat(title_in,' p-refinement');
% title(title_str);
% filename=strcat(save_in,'_p-ref.fig');
% % saveas(h2,filename,'fig');
% close(h2);
end
    
        
    
